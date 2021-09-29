#include "cmd_args.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <regex>
#include <string>

Args
parseArgs (int argc, char *argv[])
{
  const std::regex opt_regex ("opt=(.*)");
  using Opt = Args::Opt;
  const std::map<std::string, Args::Opt> opt_map{ { "cache", Opt::Cache },
                                                  { "simd", Opt::SIMD },
                                                  { "openmp", Opt::OpenMP },
                                                  { "opencl", Opt::OpenCL } };

  Args args;

  for (int i = 1; i < argc; ++i)
    {
      std::smatch match;
      std::string arg (argv[i]);
      if (std::regex_match (arg, match, opt_regex) && match.size () == 2)
        {
          auto s = match[1].str ();
          const std::string delimiter = ",";
          std::size_t start = 0;
          std::size_t end = s.find (delimiter, start);
          do
            {
              auto opt = s.substr (start, end - start);
              start = end + 1;
              args.optimization_mode
                  |= static_cast<std::uint8_t> (opt_map.at (opt));

              start = end + 1;
            }
          while ((end = s.find (delimiter, start)) != std::string::npos);
        }
    }
  return args;
}
