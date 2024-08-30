// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_EPIC
#define MFEM_EPIC

#include "../config/config.hpp"

#ifdef MFEM_USE_EPIC

// SUNDIALS vectors
#include <nvector/nvector_serial.h>
#ifdef MFEM_USE_MPI
#include <mpi.h>
#include <nvector/nvector_parallel.h>
#endif

#include "ode.hpp"
#include "solvers.hpp"
#include <Epic.h>

//Jau-Uei: to use SundialsNVector
#include "sundials.hpp"

namespace mfem
{

typedef void (*JacobianFun)(const realtype t, const Vector &y, const Vector& v, Vector& Jv, void* user_data);
// ---------------------------------------------------------------------------
// Interface to the EPIC library -- exponential methods
// ---------------------------------------------------------------------------

class EPICSolver : public ODESolver
{
protected:
	long saved_global_size;    ///< Global vector length on last initialization.

    EPICNumJacDelta Delta;
    Operator* Jtv;
    SundialsNVector* temp;
    int m[2];
    int m_max;
    int m_tmp[2];
    double kry_tol;
    bool exactJacobian;
    int myProc;
    bool printinfo;
    bool useKiops;
    int  numBand;

#ifdef MFEM_USE_MPI
    bool Parallel() const
    {
    	return (temp->MPIPlusX() || temp->GetNVectorID() == SUNDIALS_NVEC_PARALLEL);
    }
#else
    bool Parallel() const { return false; }
#endif

public:
    EPICSolver(bool exactJacobian, EPICNumJacDelta delta=&DefaultDelta);

#ifdef MFEM_USE_MPI
    EPICSolver(MPI_Comm comm, bool exactJacobian, int numBand_=0, bool useKiops_=true, bool printinfo_=false, EPICNumJacDelta delta=&DefaultDelta);
#endif

    static int RHS(realtype t, const N_Vector y, N_Vector ydot, void *user_data);
    static int Jacobian(N_Vector v, N_Vector Jv, realtype t,
                         N_Vector y, N_Vector fy, void *user_data, N_Vector tmp);
    virtual void Init(TimeDependentOperator &f, int* m_, double kry_tol_, int m_max_);
    virtual void Step(Vector &x, double &t, double &dt);

    /// Set the linear Operator of the system and initialize Jtv. (Copied from "KINSOL")
    /** @note If this method is called a second time with a different problem
        size, then non-default KINSOL-specific options will be lost and will need
        to be set again. */
    virtual void SetOperator(Operator &op);

    virtual ~EPICSolver() {}
};

//class EPI2 : public EPICSolver
//{
//protected:
//    Epi2_KIOPS* integrator;
//public:
//    EPI2(bool exactJacobian=true, EPICNumJacDelta delta=&DefaultDelta);
//
//#ifdef MFEM_USE_MPI
//    EPI2(MPI_Comm comm, bool exactJacobian=true, EPICNumJacDelta delta=&DefaultDelta);
//#endif
//
//    virtual void Init(TimeDependentOperator &f, int* m_, double kry_tol_, int m_max_);
//    virtual void Step(Vector &x, double &t, double &dt);
//
//    virtual ~EPI2();
//};

// This is used to print out information within the EPIC library.
class EPI2_debug : public EPICSolver
{
protected:
    Epi2_KIOPS_debug* integrator;
public:
    EPI2_debug(bool exactJacobian=true, EPICNumJacDelta delta=&DefaultDelta);

#ifdef MFEM_USE_MPI
    EPI2_debug(MPI_Comm comm, bool exactJacobian=true, int numBand_=0, bool useKiops_=true, bool printinfo_=false, EPICNumJacDelta delta=&DefaultDelta);
#endif

    virtual void Init(TimeDependentOperator &f, int* m_, double kry_tol_, int m_max_);
    virtual void Step(Vector &x, double &t, double &dt);

    virtual ~EPI2_debug();
};


class EPIRB32 : public EPICSolver
{
protected:
	EpiRB32_KIOPS* integrator;
public:
	EPIRB32(bool exactJacobian=true, EPICNumJacDelta delta=&DefaultDelta);

#ifdef MFEM_USE_MPI
	EPIRB32(MPI_Comm comm, bool exactJacobian=true,  int numBand_=0, bool useKiops_=true, bool printinfo_=false, EPICNumJacDelta delta=&DefaultDelta);
#endif

    virtual void Init(TimeDependentOperator &f, int* m_, double kry_tol_, int m_max_);
    virtual void Step(Vector &x, double &t, double &dt);

    virtual ~EPIRB32();
};

class EPIRB43 : public EPICSolver
{
protected:
	EpiRB43_KIOPS* integrator;
public:
	EPIRB43(bool exactJacobian=true, EPICNumJacDelta delta=&DefaultDelta);

#ifdef MFEM_USE_MPI
	EPIRB43(MPI_Comm comm, bool exactJacobian=true,  int numBand_=0, bool useKiops_=true, bool printinfo_=false, EPICNumJacDelta delta=&DefaultDelta);
#endif

    virtual void Init(TimeDependentOperator &f, int* m_, double kry_tol_, int m_max_);
    virtual void Step(Vector &x, double &t, double &dt);

    virtual ~EPIRB43();
};

//class EPIRK4 : public EPICSolver
//{
//protected:
//    EpiRK4SC_KIOPS* integrator;
//public:
//    EPIRK4(bool exactJacobian=true, EPICNumJacDelta delta=&DefaultDelta);
//
//#ifdef MFEM_USE_MPI
//    EPIRK4(MPI_Comm comm, bool exactJacobian=true, EPICNumJacDelta delta=&DefaultDelta);
//#endif
//
//    virtual void Init(TimeDependentOperator &f);
//    virtual void Step(Vector &x, double &t, double &dt);
//
//    virtual ~EPIRK4();
//};

} // namespace mfem

#endif // MFEM_USE_EPIC

#endif // MFEM_EPIC
