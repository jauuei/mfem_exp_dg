# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Defines the following variables:
#   - EPIC_FOUND
#   - EPIC_LIBRARIES
#   - EPIC_INCLUDE_DIRS

include(MfemCmakeUtilities)
mfem_find_package(EPIC EPIC EPIC_DIR
       "include" Epic.h "lib" epic1.0.0
  "Paths to headers required by EPIC." "Libraries required by EPIC.")