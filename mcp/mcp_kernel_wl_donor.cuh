#pragma once
#include "../include/defs.h"
#include "../include/queue.cuh"
#include "../include/utils.cuh"
#include "../mce/mce_utils.cuh"
#include "../mce/parameter.cuh"
#include "../mcp/mcp_utils.cuh"

// //////////////////////////////////////////////////////////////////////
// First level independent subtree with San Segundo's coloring Algorithm
// //////////////////////////////////////////////////////////////////////

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
//__launch_bounds__(BLOCK_DIM_X, 2048 / BLOCK_DIM_X)
__global__ void mcp_kernel_l1_wl_donor_psanse(
	mcp::GLOBAL_HANDLE<T> gh,
	queue_callee(queue, tickets, head, tail))
{
	mcp::LOCAL_HANDLE<T> lh;
  __shared__ mcp::SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> sh;
	__shared__ T cut_by_kcore_l1;
	__shared__ T cut_by_color;
	__shared__ T cut_by_color_l1;
	__shared__ uint32_t subgraph_edges;
	__shared__ float avg_subgraph_density;
	__shared__ float max_subgraph_density;
	__shared__ uint32_t avg_subgraph_width;
	__shared__ uint32_t max_subgraph_width;
	__shared__ uint32_t number_of_subgraph;
	__shared__ volatile unsigned long long branches;

	lh.numPartitions = BLOCK_DIM_X / CPARTSIZE; // Number of warp scheduler in block
	lh.wx = threadIdx.x / CPARTSIZE; // Warp index
	lh.lx = threadIdx.x % CPARTSIZE; // Lane index (index inside the warp)
	lh.partMask = (CPARTSIZE == 32 ? 0xFFFFFFFF : (1 << CPARTSIZE) - 1) 
					<< ((lh.wx % (32 / CPARTSIZE)) * CPARTSIZE);

	if (threadIdx.x == 0)
	{
		sh.root_sm_block_id = sh.sm_block_id = blockIdx.x;
		sh.state = 0;
		cut_by_color_l1 = 0;
		cut_by_kcore_l1 = 0;
		cut_by_color = 0;
		avg_subgraph_density = 0;
		max_subgraph_density = 0;
		avg_subgraph_width = 0;
		max_subgraph_width = 0;
		number_of_subgraph = 0;
		branches = 0;
	}
	__syncthreads();

	while (sh.state != 100)
	{

		__syncthreads();
		// If First Level or begin
		if (sh.state == 0)
		{
			if (threadIdx.x == 0) {
				// Each block gets the vertex
				//printf("block %u extracts a candidate level 1\n", blockIdx.x);
				sh.i = atomicAdd((T *)gh.work_stealing, gh.stride);
				//printf("branching on node: %d\n", sh.i);  
			}
			__syncthreads();

			// If work finished go in the waiting queue
			if (sh.i >= gh.iteration_limit) // L1 ? #Nodes : #Edges
			{
				__syncthreads();
				if (threadIdx.x == 0)
				{
					//printf("First level terminated!\n");
					sh.state = 1; // Wait state
					queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
				}
				__syncthreads();
				continue;
			}

			if (gh.core[sh.i] + 1 <= (*gh.Cmax_size))
			{
				__syncthreads();
				continue;
			}
			mcp::setup_stack_first_level_psanse(gh, sh);

			if (sh.usrcLen <= (*gh.Cmax_size))
			{
				__syncthreads();
				continue;
			}

			mcp::encode_clear(lh, sh, sh.usrcLen);
		
			// Compute the induced subgraph for kcore first level
			for (T j = lh.wx; j < sh.usrcLen; j += lh.numPartitions)
			{
				auto &g = gh.gsplit;
				graph::warp_sorted_count_and_encode_full<T, CPARTSIZE>(
					&g.colInd[sh.usrcStart], sh.usrcLen,
					&g.colInd[g.rowPtr[g.colInd[sh.usrcStart + j]]], 
					g.splitPtr[g.colInd[sh.usrcStart + j]] 
					- g.rowPtr[g.colInd[sh.usrcStart + j]],
					j, sh.num_divs_local, sh.encode);
			}

			__syncthreads();

			// Run greedy coloring
			const T ub = (*gh.Cmax_size);
			int xi = mcp::compute_upperbound_chromatic_number_psanse(lh, sh, ub - 1, sh.num_divs_local);
			if (xi <= ub - 1)
			{
				if (threadIdx.x == 0 && gh.verbose) cut_by_color_l1++;
				__syncthreads();
				continue;
			}

			// Compute k-core reduction of the first level and the ordering
			mcp::compute_kcore_first_level(lh, sh, gh, sh.num_divs_local, sh.encode);

			if (sh.usrcLen <= (*gh.Cmax_size) || sh.max_core_l1 + 1 <= (*gh.Cmax_size)) {
				if (threadIdx.x == 0 && gh.verbose)	cut_by_kcore_l1++;
				__syncthreads();
				continue;
			}

			mcp::reduce_stack_first_level(gh, sh);
			mcp::encode_clear(lh, sh, sh.usrcLen);
			mcp::reverse_degeneracy_ordering_first_level(lh, sh, gh);

			// Compute the induced subgraph based on degeneracy ordered first level
			// Warp-parallel
			if (threadIdx.x == 0)
				subgraph_edges = 0;

			__syncthreads();

			if (gh.verbose)
			{
				for (T j = lh.wx; j < sh.usrcLen; j += lh.numPartitions)
				{
					auto &g = gh.gsplit;
					graph::warp_sorted_count_and_encode_full_stats<T, CPARTSIZE>(
						sh.ordering, sh.usrcLen,
						&g.colInd[g.rowPtr[sh.ordering[j]]], 
						g.splitPtr[sh.ordering[j]] 
						- g.rowPtr[sh.ordering[j]],
						j, sh.num_divs_local, sh.encode, subgraph_edges);
				}
			}
			else
			{
				for (T j = lh.wx; j < sh.usrcLen; j += lh.numPartitions)
				{
					auto &g = gh.gsplit;
					graph::warp_sorted_count_and_encode_full<T, CPARTSIZE>(
						sh.ordering, sh.usrcLen,
						&g.colInd[g.rowPtr[sh.ordering[j]]], 
						g.splitPtr[sh.ordering[j]] 
						- g.rowPtr[sh.ordering[j]],
						j, sh.num_divs_local, sh.encode);
				}
			}
			
			__syncthreads();

			if (threadIdx.x == 0 && gh.verbose)
			{
				float current_density = static_cast<float>(subgraph_edges) / static_cast<float>(sh.usrcLen * (sh.usrcLen - 1));
				max_subgraph_density = max(max_subgraph_density, current_density);
				max_subgraph_width = max(max_subgraph_width, sh.usrcLen);
				avg_subgraph_width += sh.usrcLen;
				avg_subgraph_density += current_density;
				number_of_subgraph++;
				//atomicAdd((unsigned long long*)&branches, 1);
			}

			__syncthreads();

			// Determine intersection and put them into sh.pl
			mcp::compute_P_intersection_to_first_level(sh, sh.usrcLen, sh.num_divs_local, sh.pl);

			if (threadIdx.x == 0 && gh.eval) sh.c[0] = sh.i;

			__syncthreads();

		}
		else if (sh.state == 1) // Wait in the queue
		{
			__syncthreads();
			if (threadIdx.x == 0)
			{
				//printf("block %d waiting\n", blockIdx.x);
				//printf("Waiting blocks: %u / %u\n", tail->load(cuda::memory_order_relaxed) - head->load(cuda::memory_order_relaxed), CB);
				wait_for_donor(gh.work_ready[sh.sm_block_id], sh.state, 
							queue_caller(queue, tickets, head, tail));
			}
			__syncthreads();
			continue;
		}
		else if (sh.state == 2) // Get work from queue at first level
		{
			__syncthreads();
			mcp::setup_stack_donor_psanse(gh, sh);
		}
		
		while (sh.l >= sh.base_l)
		{
			__syncthreads();
			
			if (!sh.colored[current_level]) 
			{
				// If P = 0
				if (!gh.eval && mcp::p_maximality(gh, lh, sh) || gh.eval && mcp::p_maximality_eval(gh, lh, sh)) continue;
				if (gh.verbose && threadIdx.x == 0) atomicAdd((unsigned long long*)&branches, 1);
				mcp::compute_branches_fast(lh, sh, int((*gh.Cmax_size)) - int(sh.l - 1), sh.num_divs_local,
					current_level, sh.pl + current_level * sh.num_divs_local);
				// If B = 0
				if (mcp::b_maximality(gh, sh, lh)) { if (threadIdx.x == 0 && gh.verbose) cut_by_color++; continue; }
				mcp::compute_branching_aux_set(sh, sh.pl + current_level * sh.num_divs_local, 
					sh.al + current_level * sh.num_divs_local, sh.bl + current_level * sh.num_divs_local);
			}

			if ((*gh.work_stealing) >= gh.iteration_limit) 
			{
				mcp::prepare_fork(sh);
				mcp::get_candidates_for_next_level<true>(lh, sh, current_level, sh.bl + current_level * sh.num_divs_local);
				mcp::try_dequeue(gh, sh, queue_caller(queue, tickets, head, tail));
				
				if (sh.fork) mcp::do_fork(gh, sh, sh.num_divs_local,
					sh.bl + current_level * sh.num_divs_local,
					 sh.al + current_level * sh.num_divs_local,
					  queue_caller(queue, tickets, head, tail));
			}

			// If B visited
			if (!mcp::get_next_branching_index(gh, lh, sh,
				sh.level_pointer_index[current_level], 
				sh.usrcLen, sh.bl + sh.num_divs_local * current_level)) continue;

			//first_thread_block printf("i: %u, c: %u, l: %u\n", sh.i, lh.newIndex, sh.l);

			mcp::next_pointer(lh, sh.level_pointer_index[current_level]);
			mcp::compute_P_intersection_for_next_level(lh, sh, sh.num_divs_local,
				sh.bl + current_level * sh.num_divs_local, sh.al + current_level * sh.num_divs_local,
				sh.pl + next_level * sh.num_divs_local);
			
			mcp::go_to_next_level(sh.l, sh.level_pointer_index[next_level], sh.colored[next_level]);

			if (threadIdx.x == 0 && gh.eval) sh.c[sh.l - 2] = sh.ordering[lh.newIndex];

			__syncthreads();
		}

		__syncthreads();
		if (threadIdx.x == 0 && sh.state == 2)
		{
			sh.state = 1;
			queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
		}
		__syncthreads();
	}
	__syncthreads();

	// Collect statistics	
	if (gh.verbose && threadIdx.x == 0)
	{
		atomicAdd((T*)gh.cut_by_kcore_l1, cut_by_kcore_l1);
		atomicAdd((T*)gh.cut_by_color_l1, cut_by_color_l1);
		atomicAdd((T*)gh.cut_by_color, cut_by_color);
		atomicAdd((uint32_t*)gh.total_subgraph, number_of_subgraph);
		atomicAdd((float*)gh.avg_subgraph_density, avg_subgraph_density);
		atomicMax((float*)gh.max_subgraph_density, max_subgraph_density);
		atomicAdd((uint32_t*)gh.avg_subgraph_width, avg_subgraph_width);
		atomicMax((uint32_t*)gh.max_subgraph_width, max_subgraph_width);
		atomicAdd((unsigned long long*)gh.branches, branches);
	}
	
	__syncthreads();
}


// ////////////////////////////////////////////////////////////////////////
// First level independent subtree with Tomita's NUMBER coloring Algorithm
// ////////////////////////////////////////////////////////////////////////

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 2048 / BLOCK_DIM_X)
__global__ void mcp_kernel_l1_wl_donor_tomita(
	mcp::GLOBAL_HANDLE<T> gh,
	queue_callee(queue, tickets, head, tail))
{
	mcp::LOCAL_HANDLE<T> lh;
  __shared__ mcp::SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> sh;
	__shared__ T cut_by_kcore_l1;
	__shared__ T cut_by_color;
	__shared__ T cut_by_color_l1;
	__shared__ uint32_t subgraph_edges;
	__shared__ float avg_subgraph_density;
	__shared__ float max_subgraph_density;
	__shared__ uint32_t avg_subgraph_width;
	__shared__ uint32_t max_subgraph_width;
	__shared__ uint32_t number_of_subgraph;

	lh.numPartitions = BLOCK_DIM_X / CPARTSIZE; // Number of warp scheduler in block
	lh.wx = threadIdx.x / CPARTSIZE; // Warp index
	lh.lx = threadIdx.x % CPARTSIZE; // Lane index (index inside the warp)
	lh.partMask = (CPARTSIZE == 32 ? 0xFFFFFFFF : (1 << CPARTSIZE) - 1) 
					<< ((lh.wx % (32 / CPARTSIZE)) * CPARTSIZE);

	if (threadIdx.x == 0)
	{
		sh.root_sm_block_id = sh.sm_block_id = blockIdx.x;
		sh.state = 0;
		cut_by_color_l1 = 0;
		cut_by_kcore_l1 = 0;
		cut_by_color = 0;
		avg_subgraph_density = 0;
		max_subgraph_density = 0;
		avg_subgraph_width = 0;
		max_subgraph_width = 0;
		number_of_subgraph = 0;
	}
	__syncthreads();

	while (sh.state != 100)
	{
		__syncthreads();
		// If First Level or begin
		if (sh.state == 0)
		{
			if (threadIdx.x == 0) {
				// Each block gets the vertex
				//printf("block %u extracts a candidate level 1\n", blockIdx.x);
				sh.i = atomicAdd((T *)gh.work_stealing, gh.stride);
				//printf("branching on node: %d\n", sh.i);  
			}
			__syncthreads();

			// If work finished go in the waiting queue
			if (sh.i >= gh.iteration_limit) // L1 ? #Nodes : #Edges
			{
				__syncthreads();
				if (threadIdx.x == 0)
				{
					//printf("First level terminated!\n");
					sh.state = 1; // Wait state
					queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
				}
				__syncthreads();
				continue;
			}

			if (gh.core[sh.i] + 1 <= (*gh.Cmax_size))
			{
				__syncthreads();
				continue;
			}
			mcp::setup_stack_first_level_tomita(gh, sh);

			if (sh.usrcLen <= (*gh.Cmax_size))
			{
				__syncthreads();
				continue;
			}

			mcp::encode_clear(lh, sh, sh.usrcLen);
		
			// Compute the induced subgraph for kcore first level
			for (T j = lh.wx; j < sh.usrcLen; j += lh.numPartitions)
			{
				auto &g = gh.gsplit;
				graph::warp_sorted_count_and_encode_full<T, CPARTSIZE>(
					&g.colInd[sh.usrcStart], sh.usrcLen,
					&g.colInd[g.rowPtr[g.colInd[sh.usrcStart + j]]], 
					g.splitPtr[g.colInd[sh.usrcStart + j]] 
					- g.rowPtr[g.colInd[sh.usrcStart + j]],
					j, sh.num_divs_local, sh.encode);
			}

			__syncthreads();

			// Run greedy coloring
			const T ub = (*gh.Cmax_size);
			int xi = mcp::compute_upperbound_chromatic_number_tomita(lh, sh, ub - 1, sh.num_divs_local);
			if (xi <= ub - 1)
			{
				if (threadIdx.x == 0 && gh.verbose) cut_by_color_l1++;
				__syncthreads();
				continue;
			}

			// Compute k-core reduction of the first level and the ordering
			mcp::compute_kcore_first_level(lh, sh, gh, sh.num_divs_local, sh.encode);

			if (sh.usrcLen <= (*gh.Cmax_size) || sh.max_core_l1 + 1 <= (*gh.Cmax_size)) {
				if (threadIdx.x == 0 && gh.verbose)	cut_by_kcore_l1++;
				__syncthreads();
				continue;
			}

			mcp::reduce_stack_first_level(gh, sh);
			mcp::encode_clear(lh, sh, sh.usrcLen);
			mcp::reverse_degeneracy_ordering_first_level(lh, sh, gh);

			// Compute the induced subgraph based on degeneracy ordered first level
			// Warp-parallel
			if (threadIdx.x == 0)
				subgraph_edges = 0;

			__syncthreads();

			if (gh.verbose)
			{
				for (T j = lh.wx; j < sh.usrcLen; j += lh.numPartitions)
				{
					auto &g = gh.gsplit;
					graph::warp_sorted_count_and_encode_full_stats<T, CPARTSIZE>(
						sh.ordering, sh.usrcLen,
						&g.colInd[g.rowPtr[sh.ordering[j]]], 
						g.splitPtr[sh.ordering[j]] 
						- g.rowPtr[sh.ordering[j]],
						j, sh.num_divs_local, sh.encode, subgraph_edges);
				}
			}
			else
			{
				for (T j = lh.wx; j < sh.usrcLen; j += lh.numPartitions)
				{
					auto &g = gh.gsplit;
					graph::warp_sorted_count_and_encode_full<T, CPARTSIZE>(
						sh.ordering, sh.usrcLen,
						&g.colInd[g.rowPtr[sh.ordering[j]]], 
						g.splitPtr[sh.ordering[j]] 
						- g.rowPtr[sh.ordering[j]],
						j, sh.num_divs_local, sh.encode);
				}
			}
			
			__syncthreads();

			if (threadIdx.x == 0 && gh.verbose)
			{
				float current_density = static_cast<float>(subgraph_edges) / static_cast<float>(sh.usrcLen * (sh.usrcLen - 1));
				max_subgraph_density = max(max_subgraph_density, current_density);
				max_subgraph_width = max(max_subgraph_width, sh.usrcLen);
				avg_subgraph_width += sh.usrcLen;
				avg_subgraph_density += current_density;
				number_of_subgraph++;
			}

			__syncthreads();

			// Determine intersection and put them into sh.pl
			mcp::compute_P_intersection_to_first_level(sh, sh.usrcLen, sh.num_divs_local, sh.pl);

		}
		else if (sh.state == 1) // Wait in the queue
		{
			__syncthreads();
			if (threadIdx.x == 0)
			{
				//printf("block %d waiting\n", blockIdx.x);
				//printf("Waiting blocks: %u / %u\n", tail->load(cuda::memory_order_relaxed) - head->load(cuda::memory_order_relaxed), CB);
				wait_for_donor(gh.work_ready[sh.sm_block_id], sh.state, 
							queue_caller(queue, tickets, head, tail));
			}
			__syncthreads();
			continue;
		}
		else if (sh.state == 2) // Get work from queue at first level
		{
			__syncthreads();
			mcp::setup_stack_donor_tomita(gh, sh);
		}
		
		while (sh.l >= sh.base_l)
		{
			__syncthreads();
			
			if (!sh.colored[current_level]) 
			{
				// If P = 0
				if (mcp::p_maximality(gh, lh, sh)) continue;
				mcp::compute_branches_number(lh, sh, int((*gh.Cmax_size)) - int(sh.l - 1),
					sh.num_divs_local, current_level, sh.pl + current_level * sh.num_divs_local);
				// If B = 0
				if (mcp::b_maximality(gh, sh, lh)) { if (threadIdx.x == 0 && gh.verbose) cut_by_color++; continue; }
				mcp::compute_branching_aux_set(sh, sh.pl + current_level * sh.num_divs_local, 
					sh.al + current_level * sh.num_divs_local, sh.bl + current_level * sh.num_divs_local);
			}

			if ((*gh.work_stealing) >= gh.iteration_limit) 
			{
				mcp::prepare_fork(sh);
				mcp::get_candidates_for_next_level<true>(lh, sh, current_level, sh.bl + current_level * sh.num_divs_local);
				mcp::try_dequeue(gh, sh, queue_caller(queue, tickets, head, tail));
				
				if (sh.fork) mcp::do_fork(gh, sh, sh.num_divs_local,
					sh.bl + current_level * sh.num_divs_local,
					 sh.al + current_level * sh.num_divs_local,
					  queue_caller(queue, tickets, head, tail));
			}

			// If B visited
			if (!mcp::get_next_branching_index(gh, lh, sh,
				sh.level_pointer_index[current_level], 
				sh.usrcLen, sh.bl + sh.num_divs_local * current_level)) continue;

			mcp::next_pointer(lh, sh.level_pointer_index[current_level]);
			mcp::compute_P_intersection_for_next_level(lh, sh, sh.num_divs_local,
				sh.bl + current_level * sh.num_divs_local, sh.al + current_level * sh.num_divs_local,
				sh.pl + next_level * sh.num_divs_local);
			
			mcp::go_to_next_level(sh.l, sh.level_pointer_index[next_level], sh.colored[next_level]);
		}

		__syncthreads();
		if (threadIdx.x == 0 && sh.state == 2)
		{
			sh.state = 1;
			queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
		}
		__syncthreads();
	}
	__syncthreads();

	// Collect statistics	
	if (gh.verbose && threadIdx.x == 0)
	{
		atomicAdd((T*)gh.cut_by_kcore_l1, cut_by_kcore_l1);
		atomicAdd((T*)gh.cut_by_color_l1, cut_by_color_l1);
		atomicAdd((T*)gh.cut_by_color, cut_by_color);
		atomicAdd((uint32_t*)gh.total_subgraph, number_of_subgraph);
		atomicAdd((float*)gh.avg_subgraph_density, avg_subgraph_density);
		atomicMax((float*)gh.max_subgraph_density, max_subgraph_density);
		atomicAdd((uint32_t*)gh.avg_subgraph_width, avg_subgraph_width);
		atomicMax((uint32_t*)gh.max_subgraph_width, max_subgraph_width);
	}

	__syncthreads();
}


// /////////////////////////////////////////////////////////////////////////
// First level independent subtree with San Segundo's Re-Coloring Algorithm
// /////////////////////////////////////////////////////////////////////////

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 2048 / BLOCK_DIM_X)
__global__ void mcp_kernel_l1_wl_donor_psanse_recolor(
	mcp::GLOBAL_HANDLE<T> gh,
	queue_callee(queue, tickets, head, tail))
{
	mcp::LOCAL_HANDLE<T> lh;
  __shared__ mcp::SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> sh;
	__shared__ T cut_by_kcore_l1;
	__shared__ T cut_by_color;
	__shared__ T cut_by_color_l1;
	__shared__ uint32_t subgraph_edges;
	__shared__ float avg_subgraph_density;
	__shared__ float max_subgraph_density;
	__shared__ uint32_t avg_subgraph_width;
	__shared__ uint32_t max_subgraph_width;
	__shared__ uint32_t number_of_subgraph;

	lh.numPartitions = BLOCK_DIM_X / CPARTSIZE; // Number of warp scheduler in block
	lh.wx = threadIdx.x / CPARTSIZE; // Warp index
	lh.lx = threadIdx.x % CPARTSIZE; // Lane index (index inside the warp)
	lh.partMask = (CPARTSIZE == 32 ? 0xFFFFFFFF : (1 << CPARTSIZE) - 1) 
					<< ((lh.wx % (32 / CPARTSIZE)) * CPARTSIZE);

	if (threadIdx.x == 0)
	{
		sh.root_sm_block_id = sh.sm_block_id = blockIdx.x;
		sh.state = 0;
		cut_by_color_l1 = 0;
		cut_by_kcore_l1 = 0;
		cut_by_color = 0;
		avg_subgraph_density = 0;
		max_subgraph_density = 0;
		avg_subgraph_width = 0;
		max_subgraph_width = 0;
		number_of_subgraph = 0;
	}
	__syncthreads();

	while (sh.state != 100)
	{
		__syncthreads();
		// If First Level or begin
		if (sh.state == 0)
		{
			if (threadIdx.x == 0) {
				// Each block gets the vertex
				//printf("block %u extracts a candidate level 1\n", blockIdx.x);
				sh.i = atomicAdd((T *)gh.work_stealing, gh.stride);
				//printf("branching on node: %d\n", sh.i);  
			}
			__syncthreads();

			// If work finished go in the waiting queue
			if (sh.i >= gh.iteration_limit) // L1 ? #Nodes : #Edges
			{
				__syncthreads();
				if (threadIdx.x == 0)
				{
					//printf("First level terminated!\n");
					sh.state = 1; // Wait state
					queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
				}
				__syncthreads();
				continue;
			}

			if (gh.core[sh.i] + 1 <= (*gh.Cmax_size))
			{
				__syncthreads();
				continue;
			}
			mcp::setup_stack_first_level_psanse_recolor(gh, sh);

			if (sh.usrcLen <= (*gh.Cmax_size))
			{
				__syncthreads();
				continue;
			}

			mcp::encode_clear(lh, sh, sh.usrcLen);
		
			// Compute the induced subgraph for kcore first level
			for (T j = lh.wx; j < sh.usrcLen; j += lh.numPartitions)
			{
				auto &g = gh.gsplit;
				graph::warp_sorted_count_and_encode_full<T, CPARTSIZE>(
					&g.colInd[sh.usrcStart], sh.usrcLen,
					&g.colInd[g.rowPtr[g.colInd[sh.usrcStart + j]]], 
					g.splitPtr[g.colInd[sh.usrcStart + j]] 
					- g.rowPtr[g.colInd[sh.usrcStart + j]],
					j, sh.num_divs_local, sh.encode);
			}

			__syncthreads();

			// Run greedy coloring
			const T ub = (*gh.Cmax_size);
			int xi = mcp::compute_upperbound_chromatic_number_psanse(lh, sh, ub - 1, sh.num_divs_local);
			if (xi <= ub - 1)
			{
				if (threadIdx.x == 0 && gh.verbose) cut_by_color_l1++;
				__syncthreads();
				continue;
			}

			// Compute k-core reduction of the first level and the ordering
			mcp::compute_kcore_first_level(lh, sh, gh, sh.num_divs_local, sh.encode);

			if (sh.usrcLen <= (*gh.Cmax_size) || sh.max_core_l1 + 1 <= (*gh.Cmax_size)) {
				if (threadIdx.x == 0 && gh.verbose)	cut_by_kcore_l1++;
				__syncthreads();
				continue;
			}

			mcp::reduce_stack_first_level(gh, sh);
			mcp::encode_clear(lh, sh, sh.usrcLen);
			mcp::reverse_degeneracy_ordering_first_level(lh, sh, gh);

			// Compute the induced subgraph based on degeneracy ordered first level
			// Warp-parallel
			if (threadIdx.x == 0)
				subgraph_edges = 0;

			__syncthreads();

			if (gh.verbose)
			{
				for (T j = lh.wx; j < sh.usrcLen; j += lh.numPartitions)
				{
					auto &g = gh.gsplit;
					graph::warp_sorted_count_and_encode_full_stats<T, CPARTSIZE>(
						sh.ordering, sh.usrcLen,
						&g.colInd[g.rowPtr[sh.ordering[j]]], 
						g.splitPtr[sh.ordering[j]] 
						- g.rowPtr[sh.ordering[j]],
						j, sh.num_divs_local, sh.encode, subgraph_edges);
				}
			}
			else
			{
				for (T j = lh.wx; j < sh.usrcLen; j += lh.numPartitions)
				{
					auto &g = gh.gsplit;
					graph::warp_sorted_count_and_encode_full<T, CPARTSIZE>(
						sh.ordering, sh.usrcLen,
						&g.colInd[g.rowPtr[sh.ordering[j]]], 
						g.splitPtr[sh.ordering[j]] 
						- g.rowPtr[sh.ordering[j]],
						j, sh.num_divs_local, sh.encode);
				}
			}
			
			__syncthreads();

			if (threadIdx.x == 0 && gh.verbose)
			{
				float current_density = static_cast<float>(subgraph_edges) / static_cast<float>(sh.usrcLen * (sh.usrcLen - 1));
				max_subgraph_density = max(max_subgraph_density, current_density);
				max_subgraph_width = max(max_subgraph_width, sh.usrcLen);
				avg_subgraph_width += sh.usrcLen;
				avg_subgraph_density += current_density;
				number_of_subgraph++;
			}

			__syncthreads();

			// Determine intersection and put them into sh.pl
			mcp::compute_P_intersection_to_first_level(sh, sh.usrcLen, sh.num_divs_local, sh.pl);

		}
		else if (sh.state == 1) // Wait in the queue
		{
			__syncthreads();
			if (threadIdx.x == 0)
			{
				//printf("block %d waiting\n", blockIdx.x);
				//printf("Waiting blocks: %u / %u\n", tail->load(cuda::memory_order_relaxed) - head->load(cuda::memory_order_relaxed), CB);
				wait_for_donor(gh.work_ready[sh.sm_block_id], sh.state, 
							queue_caller(queue, tickets, head, tail));
			}
			__syncthreads();
			continue;
		}
		else if (sh.state == 2) // Get work from queue at first level
		{
			__syncthreads();
			mcp::setup_stack_donor_psanse_recolor(gh, sh);
		}
		
		while (sh.l >= sh.base_l)
		{
			__syncthreads();
			
			if (!sh.colored[current_level]) 
			{
				// If P = 0
				if (mcp::p_maximality(gh, lh, sh)) continue;
				mcp::compute_branches_fast_recolor(lh, sh, int((*gh.Cmax_size)) - int(sh.l - 1),
					sh.num_divs_local, sh.pl + current_level * sh.num_divs_local);
				// If B = 0
				if (mcp::b_maximality(gh, sh, lh)) { if (threadIdx.x == 0 && gh.verbose) cut_by_color++; continue; }
				mcp::compute_branching_aux_set(sh, sh.pl + current_level * sh.num_divs_local,
					sh.al + current_level * sh.num_divs_local, sh.bl + current_level * sh.num_divs_local);
			}

			if ((*gh.work_stealing) >= gh.iteration_limit) 
			{
				mcp::prepare_fork(sh);
				mcp::get_candidates_for_next_level<true>(lh, sh, current_level, sh.bl + current_level * sh.num_divs_local);
				mcp::try_dequeue(gh, sh, queue_caller(queue, tickets, head, tail));
				
				if (sh.fork) mcp::do_fork(gh, sh, sh.num_divs_local,
					sh.bl + current_level * sh.num_divs_local,
					 sh.al + current_level * sh.num_divs_local,
					  queue_caller(queue, tickets, head, tail));
			}

			// If B visited
			if (!mcp::get_next_branching_index(gh, lh, sh,
				sh.level_pointer_index[current_level], 
				sh.usrcLen, sh.bl + sh.num_divs_local * current_level)) continue;

			mcp::next_pointer(lh, sh.level_pointer_index[current_level]);
			mcp::compute_P_intersection_for_next_level(lh, sh, sh.num_divs_local,
				sh.bl + current_level * sh.num_divs_local, sh.al + current_level * sh.num_divs_local,
				sh.pl + next_level * sh.num_divs_local);
			
			mcp::go_to_next_level(sh.l, sh.level_pointer_index[next_level], sh.colored[next_level]);
		}

		__syncthreads();
		if (threadIdx.x == 0 && sh.state == 2)
		{
			sh.state = 1;
			queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
		}
		__syncthreads();
	}
	__syncthreads();

	// Collect statistics	
	if (gh.verbose && threadIdx.x == 0)
	{
		atomicAdd((T*)gh.cut_by_kcore_l1, cut_by_kcore_l1);
		atomicAdd((T*)gh.cut_by_color_l1, cut_by_color_l1);
		atomicAdd((T*)gh.cut_by_color, cut_by_color);
		atomicAdd((uint32_t*)gh.total_subgraph, number_of_subgraph);
		atomicAdd((float*)gh.avg_subgraph_density, avg_subgraph_density);
		atomicMax((float*)gh.max_subgraph_density, max_subgraph_density);
		atomicAdd((uint32_t*)gh.avg_subgraph_width, avg_subgraph_width);
		atomicMax((uint32_t*)gh.max_subgraph_width, max_subgraph_width);
	}

	__syncthreads();
}


// //////////////////////////////////////////////////////////////////////////
// First level independent subtree with Tomita's Re-NUMBER coloring Algorithm
// //////////////////////////////////////////////////////////////////////////

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 2048 / BLOCK_DIM_X)
__global__ void mcp_kernel_l1_wl_donor_tomita_renumber(
	mcp::GLOBAL_HANDLE<T> gh,
	queue_callee(queue, tickets, head, tail))
{
	mcp::LOCAL_HANDLE<T> lh;
  __shared__ mcp::SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> sh;
	__shared__ T cut_by_kcore_l1;
	__shared__ T cut_by_color;
	__shared__ T cut_by_color_l1;
	__shared__ uint32_t subgraph_edges;
	__shared__ float avg_subgraph_density;
	__shared__ float max_subgraph_density;
	__shared__ uint32_t avg_subgraph_width;
	__shared__ uint32_t max_subgraph_width;
	__shared__ uint32_t number_of_subgraph;

	lh.numPartitions = BLOCK_DIM_X / CPARTSIZE; // Number of warp scheduler in block
	lh.wx = threadIdx.x / CPARTSIZE; // Warp index
	lh.lx = threadIdx.x % CPARTSIZE; // Lane index (index inside the warp)
	lh.partMask = (CPARTSIZE == 32 ? 0xFFFFFFFF : (1 << CPARTSIZE) - 1) 
					<< ((lh.wx % (32 / CPARTSIZE)) * CPARTSIZE);

	if (threadIdx.x == 0)
	{
		sh.root_sm_block_id = sh.sm_block_id = blockIdx.x;
		sh.state = 0;
		cut_by_color_l1 = 0;
		cut_by_kcore_l1 = 0;
		cut_by_color = 0;
		avg_subgraph_density = 0;
		max_subgraph_density = 0;
		avg_subgraph_width = 0;
		max_subgraph_width = 0;
		number_of_subgraph = 0;
	}
	__syncthreads();

	while (sh.state != 100)
	{
		__syncthreads();
		// If First Level or begin
		if (sh.state == 0)
		{
			if (threadIdx.x == 0) {
				// Each block gets the vertex
				//printf("block %u extracts a candidate level 1\n", blockIdx.x);
				sh.i = atomicAdd((T *)gh.work_stealing, gh.stride);
				//printf("branching on node: %d\n", sh.i);  
			}
			__syncthreads();

			// If work finished go in the waiting queue
			if (sh.i >= gh.iteration_limit) // L1 ? #Nodes : #Edges
			{
				__syncthreads();
				if (threadIdx.x == 0)
				{
					//printf("First level terminated!\n");
					sh.state = 1; // Wait state
					queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
				}
				__syncthreads();
				continue;
			}

			if (gh.core[sh.i] + 1 <= (*gh.Cmax_size))
			{
				__syncthreads();
				continue;
			}
			mcp::setup_stack_first_level_tomita(gh, sh);

			if (sh.usrcLen <= (*gh.Cmax_size))
			{
				__syncthreads();
				continue;
			}

			mcp::encode_clear(lh, sh, sh.usrcLen);
		
			// Compute the induced subgraph for kcore first level
			for (T j = lh.wx; j < sh.usrcLen; j += lh.numPartitions)
			{
				auto &g = gh.gsplit;
				graph::warp_sorted_count_and_encode_full<T, CPARTSIZE>(
					&g.colInd[sh.usrcStart], sh.usrcLen,
					&g.colInd[g.rowPtr[g.colInd[sh.usrcStart + j]]], 
					g.splitPtr[g.colInd[sh.usrcStart + j]] 
					- g.rowPtr[g.colInd[sh.usrcStart + j]],
					j, sh.num_divs_local, sh.encode);
			}

			__syncthreads();

			// Run greedy coloring
			const T ub = (*gh.Cmax_size) - 1;
			int xi = mcp::compute_upperbound_chromatic_number_tomita_renumber(lh, sh, ub, sh.num_divs_local);
			if (xi <= ub)
			{
				if (threadIdx.x == 0 && gh.verbose) cut_by_color_l1++;
				__syncthreads();
				continue;
			}

			// Compute k-core reduction of the first level and the ordering
			mcp::compute_kcore_first_level(lh, sh, gh, sh.num_divs_local, sh.encode);

			if (sh.usrcLen <= (*gh.Cmax_size) || sh.max_core_l1 + 1 <= (*gh.Cmax_size)) {
				if (threadIdx.x == 0 && gh.verbose)	cut_by_kcore_l1++;
				__syncthreads();
				continue;
			}

			mcp::reduce_stack_first_level(gh, sh);
			mcp::encode_clear(lh, sh, sh.usrcLen);
			mcp::reverse_degeneracy_ordering_first_level(lh, sh, gh);

			// Compute the induced subgraph based on degeneracy ordered first level
			// Warp-parallel
			if (threadIdx.x == 0)
				subgraph_edges = 0;

			__syncthreads();

			if (gh.verbose)
			{
				for (T j = lh.wx; j < sh.usrcLen; j += lh.numPartitions)
				{
					auto &g = gh.gsplit;
					graph::warp_sorted_count_and_encode_full_stats<T, CPARTSIZE>(
						sh.ordering, sh.usrcLen,
						&g.colInd[g.rowPtr[sh.ordering[j]]], 
						g.splitPtr[sh.ordering[j]] 
						- g.rowPtr[sh.ordering[j]],
						j, sh.num_divs_local, sh.encode, subgraph_edges);
				}
			}
			else
			{
				for (T j = lh.wx; j < sh.usrcLen; j += lh.numPartitions)
				{
					auto &g = gh.gsplit;
					graph::warp_sorted_count_and_encode_full<T, CPARTSIZE>(
						sh.ordering, sh.usrcLen,
						&g.colInd[g.rowPtr[sh.ordering[j]]], 
						g.splitPtr[sh.ordering[j]] 
						- g.rowPtr[sh.ordering[j]],
						j, sh.num_divs_local, sh.encode);
				}
			}
			
			__syncthreads();

			if (threadIdx.x == 0 && gh.verbose)
			{
				float current_density = static_cast<float>(subgraph_edges) / static_cast<float>(sh.usrcLen * (sh.usrcLen - 1));
				max_subgraph_density = max(max_subgraph_density, current_density);
				max_subgraph_width = max(max_subgraph_width, sh.usrcLen);
				avg_subgraph_width += sh.usrcLen;
				avg_subgraph_density += current_density;
				number_of_subgraph++;
			}

			__syncthreads();

			// Determine intersection and put them into sh.pl
			mcp::compute_P_intersection_to_first_level(sh, sh.usrcLen, sh.num_divs_local, sh.pl);

		}
		else if (sh.state == 1) // Wait in the queue
		{
			__syncthreads();
			if (threadIdx.x == 0)
			{
				//printf("block %d waiting\n", blockIdx.x);
				//printf("Waiting blocks: %u / %u\n", tail->load(cuda::memory_order_relaxed) - head->load(cuda::memory_order_relaxed), CB);
				wait_for_donor(gh.work_ready[sh.sm_block_id], sh.state, 
							queue_caller(queue, tickets, head, tail));
			}
			__syncthreads();
			continue;
		}
		else if (sh.state == 2) // Get work from queue at first level
		{
			__syncthreads();
			mcp::setup_stack_donor_tomita(gh, sh);
		}
		
		while (sh.l >= sh.base_l)
		{
			__syncthreads();
			
			if (!sh.colored[current_level]) 
			{
				// If P = 0
				if (mcp::p_maximality(gh, lh, sh)) continue;
				mcp::compute_branches_renumber(lh, sh, int((*gh.Cmax_size)) - int(sh.l - 1),
					sh.num_divs_local, current_level, sh.pl + current_level * sh.num_divs_local);
				// If B = 0
				if (mcp::b_maximality(gh, sh, lh)) { if (threadIdx.x == 0 && gh.verbose) cut_by_color++; continue; }
				mcp::compute_branching_aux_set(sh, sh.pl + current_level * sh.num_divs_local, 
					sh.al + current_level * sh.num_divs_local, sh.bl + current_level * sh.num_divs_local);
			}

			if ((*gh.work_stealing) >= gh.iteration_limit) 
			{
				mcp::prepare_fork(sh);
				mcp::get_candidates_for_next_level<true>(lh, sh, current_level, sh.bl + current_level * sh.num_divs_local);
				mcp::try_dequeue(gh, sh, queue_caller(queue, tickets, head, tail));
				
				if (sh.fork) mcp::do_fork(gh, sh, sh.num_divs_local,
					sh.bl + current_level * sh.num_divs_local,
					 sh.al + current_level * sh.num_divs_local,
					  queue_caller(queue, tickets, head, tail));
			}

			// If B visited
			if (!mcp::get_next_branching_index(gh, lh, sh,
				sh.level_pointer_index[current_level], 
				sh.usrcLen, sh.bl + sh.num_divs_local * current_level)) continue;

			mcp::next_pointer(lh, sh.level_pointer_index[current_level]);
			mcp::compute_P_intersection_for_next_level(lh, sh, sh.num_divs_local,
				sh.bl + current_level * sh.num_divs_local, sh.al + current_level * sh.num_divs_local,
				sh.pl + next_level * sh.num_divs_local);
			
			mcp::go_to_next_level(sh.l, sh.level_pointer_index[next_level], sh.colored[next_level]);
		}

		__syncthreads();
		if (threadIdx.x == 0 && sh.state == 2)
		{
			sh.state = 1;
			queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
		}
		__syncthreads();
	}
	__syncthreads();

	// Collect statistics	
	if (gh.verbose && threadIdx.x == 0)
	{
		atomicAdd((T*)gh.cut_by_kcore_l1, cut_by_kcore_l1);
		atomicAdd((T*)gh.cut_by_color_l1, cut_by_color_l1);
		atomicAdd((T*)gh.cut_by_color, cut_by_color);
		atomicAdd((uint32_t*)gh.total_subgraph, number_of_subgraph);
		atomicAdd((float*)gh.avg_subgraph_density, avg_subgraph_density);
		atomicMax((float*)gh.max_subgraph_density, max_subgraph_density);
		atomicAdd((uint32_t*)gh.avg_subgraph_width, avg_subgraph_width);
		atomicMax((uint32_t*)gh.max_subgraph_width, max_subgraph_width);
	}

	__syncthreads();
}


// /////////////////////////////////////////////////////////////////////////////
// First level independent subtree with coloring and reduce MC-BRB Optimization
// /////////////////////////////////////////////////////////////////////////////

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 2048 / BLOCK_DIM_X)
__global__ void mcp_kernel_l1_wl_donor_reduce(
	mcp::GLOBAL_HANDLE<T> gh,
	queue_callee(queue, tickets, head, tail))
{
	mcp::LOCAL_HANDLE<T> lh;
  __shared__ mcp::SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> sh;
	__shared__ T cut_by_kcore_l1;
	__shared__ T cut_by_color;
	__shared__ T cut_by_color_l1;
	__shared__ uint32_t subgraph_edges;
	__shared__ float avg_subgraph_density;
	__shared__ float max_subgraph_density;
	__shared__ uint32_t avg_subgraph_width;
	__shared__ uint32_t max_subgraph_width;
	__shared__ uint32_t number_of_subgraph;

	lh.numPartitions = BLOCK_DIM_X / CPARTSIZE; // Number of warp scheduler in block
	lh.wx = threadIdx.x / CPARTSIZE; // Warp index
	lh.lx = threadIdx.x % CPARTSIZE; // Lane index (index inside the warp)
	lh.partMask = (CPARTSIZE == 32 ? 0xFFFFFFFF : (1 << CPARTSIZE) - 1) 
					<< ((lh.wx % (32 / CPARTSIZE)) * CPARTSIZE);

	if (threadIdx.x == 0)
	{
		sh.root_sm_block_id = sh.sm_block_id = blockIdx.x;
		sh.state = 0;
		cut_by_color_l1 = 0;
		cut_by_kcore_l1 = 0;
		cut_by_color = 0;
		avg_subgraph_density = 0;
		max_subgraph_density = 0;
		avg_subgraph_width = 0;
		max_subgraph_width = 0;
		number_of_subgraph = 0;
	}
	__syncthreads();

	while (sh.state != 100)
	{
		__syncthreads();
		// If First Level or begin
		if (sh.state == 0)
		{
			if (threadIdx.x == 0) {
				// Each block gets the vertex
				//printf("block %u extracts a candidate level 1\n", blockIdx.x);
				sh.i = atomicAdd((T *)gh.work_stealing, gh.stride);
				//printf("branching on node: %d\n", sh.i);  
			}
			__syncthreads();

			// If work finished go in the waiting queue
			if (sh.i >= gh.iteration_limit) // L1 ? #Nodes : #Edges
			{
				__syncthreads();
				if (threadIdx.x == 0)
				{
					//printf("First level terminated!\n");
					sh.state = 1; // Wait state
					queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
				}
				__syncthreads();
				continue;
			}

			if (gh.core[sh.i] + 1 <= (*gh.Cmax_size))
			{
				__syncthreads();
				continue;
			}
			mcp::setup_stack_first_level_psanse(gh, sh);

			if (sh.usrcLen <= (*gh.Cmax_size))
			{
				__syncthreads();
				continue;
			}

			mcp::encode_clear(lh, sh, sh.usrcLen);
		
			// Compute the induced subgraph for kcore first level
			for (T j = lh.wx; j < sh.usrcLen; j += lh.numPartitions)
			{
				auto &g = gh.gsplit;
				graph::warp_sorted_count_and_encode_full<T, CPARTSIZE>(
					&g.colInd[sh.usrcStart], sh.usrcLen,
					&g.colInd[g.rowPtr[g.colInd[sh.usrcStart + j]]], 
					g.splitPtr[g.colInd[sh.usrcStart + j]] 
					- g.rowPtr[g.colInd[sh.usrcStart + j]],
					j, sh.num_divs_local, sh.encode);
			}

			__syncthreads();

			// Run greedy coloring
			const T ub = (*gh.Cmax_size);
			int xi = mcp::compute_upperbound_chromatic_number_psanse(lh, sh, ub - 1, sh.num_divs_local);
			if (xi <= ub - 1)
			{
				if (threadIdx.x == 0 && gh.verbose) cut_by_color_l1++;
				__syncthreads();
				continue;
			}

			// Compute k-core reduction of the first level and the ordering
			mcp::compute_kcore_first_level(lh, sh, gh, sh.num_divs_local, sh.encode);

			if (sh.usrcLen <= (*gh.Cmax_size) || sh.max_core_l1 + 1 <= (*gh.Cmax_size)) {
				if (threadIdx.x == 0 && gh.verbose)	cut_by_kcore_l1++;
				__syncthreads();
				continue;
			}

			mcp::reduce_stack_first_level(gh, sh);
			mcp::encode_clear(lh, sh, sh.usrcLen);
			mcp::reverse_degeneracy_ordering_first_level(lh, sh, gh);

			// Compute the induced subgraph based on degeneracy ordered first level
			// Warp-parallel
			if (threadIdx.x == 0)
				subgraph_edges = 0;

			__syncthreads();

			if (gh.verbose)
			{
				for (T j = lh.wx; j < sh.usrcLen; j += lh.numPartitions)
				{
					auto &g = gh.gsplit;
					graph::warp_sorted_count_and_encode_full_stats<T, CPARTSIZE>(
						sh.ordering, sh.usrcLen,
						&g.colInd[g.rowPtr[sh.ordering[j]]], 
						g.splitPtr[sh.ordering[j]] 
						- g.rowPtr[sh.ordering[j]],
						j, sh.num_divs_local, sh.encode, subgraph_edges);
				}
			}
			else
			{
				for (T j = lh.wx; j < sh.usrcLen; j += lh.numPartitions)
				{
					auto &g = gh.gsplit;
					graph::warp_sorted_count_and_encode_full<T, CPARTSIZE>(
						sh.ordering, sh.usrcLen,
						&g.colInd[g.rowPtr[sh.ordering[j]]], 
						g.splitPtr[sh.ordering[j]] 
						- g.rowPtr[sh.ordering[j]],
						j, sh.num_divs_local, sh.encode);
				}
			}
			
			__syncthreads();

			if (threadIdx.x == 0 && gh.verbose)
			{
				float current_density = static_cast<float>(subgraph_edges) / static_cast<float>(sh.usrcLen * (sh.usrcLen - 1));
				max_subgraph_density = max(max_subgraph_density, current_density);
				max_subgraph_width = max(max_subgraph_width, sh.usrcLen);
				avg_subgraph_width += sh.usrcLen;
				avg_subgraph_density += current_density;
				number_of_subgraph++;
			}

			__syncthreads();

			
			mcp::compute_P_intersection_to_first_level(sh, sh.usrcLen, sh.num_divs_local, sh.pl);

			__syncthreads();

		}
		else if (sh.state == 1) // Wait in the queue
		{
			__syncthreads();
			if (threadIdx.x == 0)
			{
				//printf("block %d waiting\n", blockIdx.x);
				//printf("Waiting blocks: %u / %u\n", tail->load(cuda::memory_order_relaxed) - head->load(cuda::memory_order_relaxed), CB);
				wait_for_donor(gh.work_ready[sh.sm_block_id], sh.state, 
							queue_caller(queue, tickets, head, tail));
			}
			__syncthreads();
			continue;
		}
		else if (sh.state == 2) // Get work from queue at first level
		{
			__syncthreads();
			mcp::setup_stack_donor_psanse(gh, sh);
		}
		
		while (sh.l >= sh.base_l)
		{
			__syncthreads();
			
			if (!sh.colored[current_level]) 
			{
				// If P = 0
				if (mcp::p_maximality(gh, lh, sh)) continue;
				if (mcp::reduce(gh, lh, sh, sh.num_divs_local, sh.l - 2, sh.pl + current_level * sh.num_divs_local)) continue;
				mcp::compute_branches_fast(lh, sh, int((*gh.Cmax_size)) - int(sh.l - 1), sh.num_divs_local,
					current_level, sh.pl + current_level * sh.num_divs_local);
				// If B = 0
				if (mcp::b_maximality(gh, sh, lh)) { if (threadIdx.x == 0 && gh.verbose) cut_by_color++; continue; }
				mcp::compute_branching_aux_set(sh, sh.pl + current_level * sh.num_divs_local, 
					sh.al + current_level * sh.num_divs_local, sh.bl + current_level * sh.num_divs_local);
			}

			if ((*gh.work_stealing) >= gh.iteration_limit) 
			{
				mcp::prepare_fork(sh);
				mcp::get_candidates_for_next_level<true>(lh, sh, current_level, sh.bl + current_level * sh.num_divs_local);
				mcp::try_dequeue(gh, sh, queue_caller(queue, tickets, head, tail));
				
				if (sh.fork) mcp::do_fork(gh, sh, sh.num_divs_local,
					sh.bl + current_level * sh.num_divs_local,
					 sh.al + current_level * sh.num_divs_local,
					  queue_caller(queue, tickets, head, tail));
			}

			// If B visited
			if (!mcp::get_next_branching_index(gh, lh, sh,
				sh.level_pointer_index[current_level], 
				sh.usrcLen, sh.bl + sh.num_divs_local * current_level)) continue;

			//first_thread_block printf("i: %u, c: %u, l: %u\n", sh.i, lh.newIndex, sh.l);

			mcp::next_pointer(lh, sh.level_pointer_index[current_level]);
			mcp::compute_P_intersection_for_next_level(lh, sh, sh.num_divs_local,
				sh.bl + current_level * sh.num_divs_local, sh.al + current_level * sh.num_divs_local,
				sh.pl + next_level * sh.num_divs_local);
			
			mcp::go_to_next_level(sh.l, sh.level_pointer_index[next_level], sh.colored[next_level]);
		}

		__syncthreads();
		if (threadIdx.x == 0 && sh.state == 2)
		{
			sh.state = 1;
			queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
		}
		__syncthreads();
	}
	__syncthreads();

	// Collect statistics	
	if (gh.verbose && threadIdx.x == 0)
	{
		atomicAdd((T*)gh.cut_by_kcore_l1, cut_by_kcore_l1);
		atomicAdd((T*)gh.cut_by_color_l1, cut_by_color_l1);
		atomicAdd((T*)gh.cut_by_color, cut_by_color);
		atomicAdd((uint32_t*)gh.total_subgraph, number_of_subgraph);
		atomicAdd((float*)gh.avg_subgraph_density, avg_subgraph_density);
		atomicMax((float*)gh.max_subgraph_density, max_subgraph_density);
		atomicAdd((uint32_t*)gh.avg_subgraph_width, avg_subgraph_width);
		atomicMax((uint32_t*)gh.max_subgraph_width, max_subgraph_width);
	}

	__syncthreads();
}
