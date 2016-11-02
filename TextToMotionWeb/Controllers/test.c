#include <stdio.h>
#include <stdint.h>

int32_t test_print_hello(char *message)
{
	printf("Hello!\n%s", message);

	return 0xDEADBEEF;
}
