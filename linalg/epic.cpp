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

#include "epic.hpp"

#ifdef MFEM_USE_EPIC

namespace mfem
{

EPICSolver::EPICSolver(bool _exactJacobian, EPICNumJacDelta _delta)
{
   saved_global_size = 0;

   // Allocate an empty serial N_Vector
   temp = new SundialsNVector();

   m[0] = 10;
   m[1] = 10;
   m_max= 100; // default max value is set to be 100;

   exactJacobian = _exactJacobian;
   Jtv = NULL;
   Delta = _delta;
}

#ifdef MFEM_USE_MPI
EPICSolver::EPICSolver(MPI_Comm _comm, bool _exactJacobian, EPICNumJacDelta _delta)
{
   saved_global_size = 0;

   m[0] = 10;
   m[1] = 10;
   m_max= 100; // default max value is set to be 100;

   // Allocate an empty parallel N_Vector
  temp = new SundialsNVector(_comm);

  exactJacobian = _exactJacobian;
  Jtv = NULL;
  Delta = _delta;
}
#endif

int EPICSolver::RHS(realtype t, const N_Vector y, N_Vector ydot, void *user_data)
{
   // Get data from N_Vectors
	// At this point the up-to-date data for N_Vector y and ydot is on the device.
   const SundialsNVector mfem_y(y);
   SundialsNVector mfem_ydot(ydot);

   EPICSolver *self = static_cast<EPICSolver*>(user_data);

   // Compute y' = f(t, y)
   self->f->SetTime(t);
   self->f->Mult(mfem_y, mfem_ydot);

   // Return success
   return 0;
}


int EPICSolver::Jacobian(N_Vector v, N_Vector Jv, realtype t, N_Vector y, N_Vector fy, void *user_data, N_Vector tmp)
{
	//N_VPrint_Parallel(v);
	//cout << endl;
	//printf("end of print N_Vector\n");

   // Similar to "GradientMult" in "KINSolver". See also how function "Jtv" is constructed in the examples of EPIK
   // Get data from N_Vectors
   const SundialsNVector mfem_v(v);
   SundialsNVector mfem_Jv(Jv);

   //mfem_v.Print(cout, 1);
   //mfem_error("test in EPICSolver::Jacobian");

   EPICSolver *self = static_cast<EPICSolver*>(user_data);

   // Compute J(t, y) v
   // XXX: "Mult" is defined in the script. (i.e., LinearOperator in MHD.cpp)
   self->Jtv->Mult(mfem_v, mfem_Jv);

   return 0;
}

void EPICSolver::SetOperator(Operator &op)
{
	// set up the computation of Jtv (linear operator)
	Jtv = &op;
}

void EPICSolver::Init(TimeDependentOperator &f, int* m_, int m_max_=0)
{
    ODESolver::Init(f);

    if (sizeof(m_)/sizeof(int)!=2)
    	mfem_error("Incorrect size of the Krylov subspace.");

    m[0] = m_[0];
    m[1] = m_[1];

    if (m_max_>0)
    	m_max=m_max_; // reset the max size the Krylov subspace size
	    
    long local_size = f.Height();
#ifdef MFEM_USE_MPI
    long global_size = 0;
    if (Parallel())
    {
        MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG, MPI_SUM,
                      temp->GetComm());
    }
#endif

    // TODO: may need "sundials_mem" (i.e., the problem size has changed since the last Init() call)
    //       when using AMR framework. See line 716 in sundials.cpp
    if (!Parallel())
    {
    	temp->SetSize(local_size);
    }
#ifdef MFEM_USE_MPI
    else
    {
    	temp->SetSize(local_size, global_size);
    	saved_global_size = global_size;
    }
#endif

}

// TODO: in the constructor, we can initialize "integrator" by nullptr. Note that it will be initialized later through the funciton "Init"
EPI2::EPI2(bool exactJacobian, EPICNumJacDelta delta) : EPICSolver(exactJacobian, delta) {}
EPI2::EPI2(MPI_Comm comm, bool exactJacobian, EPICNumJacDelta delta) : EPICSolver(comm, exactJacobian, delta) {}

void EPI2::Init(TimeDependentOperator &f, int *m_, int m_max_=0)
{
    EPICSolver::Init(f, m_, m_max_);
    long local_size = f.Height();
    long vec_size=(saved_global_size==0?local_size:saved_global_size);
    if (exactJacobian) {
       //integrator = new Epi2_KIOPS(EPICSolver::RHS, EPICSolver::Jacobian, this, 100, *temp ,vec_size);
       integrator = new Epi2_KIOPS(EPICSolver::RHS, EPICSolver::Jacobian, this, m_max, *temp ,vec_size);
    } else {
       integrator = new Epi2_KIOPS(EPICSolver::RHS, Delta, this, 100, *temp ,vec_size);
    }

}

// TODO: in the constructor, we can initialize "integrator" by nullptr. Note that it will be initialized later through the funciton "Init"
EPIRK4::EPIRK4(bool exactJacobian, EPICNumJacDelta delta) : EPICSolver(exactJacobian, delta) {}
EPIRK4::EPIRK4(MPI_Comm comm, bool exactJacobian, EPICNumJacDelta delta) : EPICSolver(comm, exactJacobian, delta) {}

void EPIRK4::Init(TimeDependentOperator &f)
{
	// TODO: need to fix Initialization
	int m_[]={10,10};
	int m_max_=100;
    EPICSolver::Init(f,&m_[0],m_max_);
    long local_size = f.Height();
    long vec_size=(saved_global_size==0?local_size:saved_global_size);
    if (exactJacobian) {
       // Here maxKrylovIters is set to be 100.
       // The minimum is always set to be 10. See M_min in Integrators/AdaptiveKrylov/Kiops.cpp in the EPIC package.
       integrator = new EpiRK4SC_KIOPS(EPICSolver::RHS, EPICSolver::Jacobian, this, 100, *temp ,vec_size);
    } else {
       integrator = new EpiRK4SC_KIOPS(EPICSolver::RHS, Delta, this, 100, *temp ,vec_size);
    }
}

void EPICSolver::Step(Vector &x, double &t, double &dt)
{
   temp->MakeRef(x, 0, x.Size());
   MFEM_VERIFY(temp->Size() == x.Size(), "size mismatch");
   // Reinitialize CVODE memory if needed (TODO: needed if we use "sundials_mem")

   // XXX:GetGradient is assumed to be implemented through SetOperator(&op)
   // i.e., see ex10p.cpp in sundials examples. The implementation is carried in
   // ReducedSystemOperator::GetGradient(const Vector &k).
//   if (exactJacobian) {
//	   Jtv = &(this->f->GetGradient(x));
//   }
}

void EPI2::Step(Vector &x, double &t, double &dt)
{
    EPICSolver::Step(x, t, dt); // update the linear operator at each time step
    //Note: dt is substep size. Currently, it is set to be single sub-time step.
    integrator->Integrate(dt, t, t+dt, 0, *temp, 1e-10, m);
    t += dt;
}

void EPIRK4::Step(Vector &x, double &t, double &dt)
{
    EPICSolver::Step(x, t, dt);
    integrator->Integrate(dt, t, t+dt, 0, *temp, 1e-10, m);
    t += dt;
}

EPI2::~EPI2()
{
	delete temp;
	delete Jtv;
    delete integrator;
}

EPIRK4::~EPIRK4()
{
	delete temp;
	delete Jtv;
    delete integrator;
}

}

#endif
