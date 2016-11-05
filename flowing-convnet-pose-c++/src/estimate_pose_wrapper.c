#include "estimate_pose_wrapper.h"

extern int32_t estimate_pose_from_c(void *image, uint32_t *size_bytes, uint32_t max_size_bytes);

int32_t estimate_pose_wrapper(void *image, uint32_t *size_bytes, uint32_t max_size_bytes)
{
	return estimate_pose_from_c(image, size_bytes, max_size_bytes);
}
