/*!
 * This kernel performs matrix multiplication
 * For now the matrices are assumed to have the same X/Y dimension n.
 */

#include "bsg_manycore.h"
#include "bsg_set_tile_x_y.h"

#include "bsg_tile_group_barrier.hpp"

bsg_barrier<bsg_tiles_X, bsg_tiles_Y> barrier;

int sys_flow_E, sys_flow_S, sys_flow_ACC;

extern "C" __attribute__ ((noinline))
int kernel_matrix_mul(int *A, int *B, int *C, int M, int N, int P, int block_size_y, int block_size_x) {


	// int start_y = __bsg_tile_group_id_y * block_size_y;
	// int start_x = __bsg_tile_group_id_x * block_size_x;
	// int end_y = start_y + block_size_y;
	// int end_x = start_x + block_size_x;
        int counter = 0;
        int counter_end = N + M + P - 2;
        int sys_flow_N = 0, sys_flow_W = 0;

        // original tiled implementation cycles: 4256

	//int end_y = M < (start_y + block_size_y) ? M : (start_y + block_size_y);
	//int end_x = P < (start_x + block_size_x) ? P : (start_x + block_size_x);

	// for (int iter_y = start_y + __bsg_y; iter_y < end_y; iter_y += bsg_tiles_Y) {
	// 	for (int iter_x = start_x + __bsg_x; iter_x < end_x; iter_x += bsg_tiles_X) {
	// 		int sum = 0;
	// 		for (int k = 0; k < N; k ++) {
	// 			sum += A[iter_y * N + k] * B[k * P + iter_x];
	// 		}
	// 		C[iter_y * P + iter_x] = sum;
	// 	}
	// }

        // systolic implementation cycles: 27273

        sys_flow_E = sys_flow_S = sys_flow_ACC = 0;

	barrier.sync();

        for (int counter = 0; counter < counter_end; counter++) {
          // fetch value from NORTH
          if (__bsg_y == 0) {
            int Bp = __bsg_x;
            int Bn = N - 1 - counter + __bsg_x;
            if (Bn >= 0 && Bn < N) {
              sys_flow_N = B[Bn * P + Bp];
            } else {
              sys_flow_N = 0;
            }
          } else {
            bsg_remote_load(__bsg_x, __bsg_y - 1, &sys_flow_S, sys_flow_N);
          }

          // fetch value from WEST
          if (__bsg_x == 0) {
            int Am = __bsg_y;
            int An = N - 1 - counter + __bsg_y;
            if (An >= 0 && An < N) {
              sys_flow_W = A[Am * N + An];
            } else {
              sys_flow_W = 0;
            }
          } else {
            bsg_remote_load(__bsg_x - 1, __bsg_y, &sys_flow_E, sys_flow_W);
          }

          sys_flow_ACC += sys_flow_N * sys_flow_W;
          //bsg_printf("ACC (%d, %d) = %d\n", __bsg_x, __bsg_y, sys_flow_ACC);

          barrier.sync();

          sys_flow_S = sys_flow_N;
          sys_flow_E = sys_flow_W;

          barrier.sync();
        }

        // if (sys_flow_ACC != C[__bsg_y * P + __bsg_x]) {
        //   bsg_fail();
        // }

        C[__bsg_y * P + __bsg_x] = sys_flow_ACC;

	barrier.sync();

	return 0;
}
