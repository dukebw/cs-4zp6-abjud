#ifndef _ESTIMATE_POSE_WRAPPER_
#define _ESTIMATE_POSE_WRAPPER_

#include <stdint.h>

int32_t estimate_pose_wrapper(void *image,
			      uint32_t size_bytes,
			      uint32_t max_size_bytes);

#endif // _ESTIMATE_POSE_WRAPPER_
