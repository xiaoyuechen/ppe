#include "cmd_args.h"

#include <argp.h>
#include <error.h>
#include <stdlib.h>

const char *argp_program_version = "cencoder 0.1";
static const char doc[] = "cencoder -- a JPEG video encoder";

#define OPT_CL_NUM_THD 1

static const struct argp_option argp_options[]
    = { { "cl", 'c', 0, 0, "Use OpenCL optimisation" },
        { "cl_num_thd", OPT_CL_NUM_THD, "NUM", 0,
          "Use NUM threads for OpenCL" },
        { "omp", 'm', 0, 0, "Use OpenMP optimisation" },
	{ "acc", 'a', 0, 0, "Use OpenACC optimisation" },
        { 0 } };

static error_t
ParseOpt (int key, char *arg, struct argp_state *state)
{
  Args *args = state->input;

  switch (key)
    {
    case 'c':
      args->optimization_mode |= OpenCL;
      break;
    case OPT_CL_NUM_THD:
      args->opencl_num_threads = strtol (arg, 0, 10);
      break;
    case 'm':
      args->optimization_mode |= OpenMP;
      break;
    case 'a':
      args->optimization_mode |= OpenACC;
      break;
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static struct argp argp = { argp_options, ParseOpt, 0, doc };

Args
parseArgs (int argc, char *argv[])
{
  Args args = { .optimization_mode = 0, .opencl_num_threads = 0 };

  argp_parse (&argp, argc, argv, 0, 0, &args);
  return args;
}
