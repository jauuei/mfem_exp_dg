// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"
#include "../general/forall.hpp"

namespace mfem
{

ParNonlinearForm::ParNonlinearForm(ParFiniteElementSpace *pf)
   : NonlinearForm(pf), pGrad(Operator::Hypre_ParCSR)
{
   X.MakeRef(pf, NULL);
   Y.MakeRef(pf, NULL);
   mX=0.0; // will not be used
   X_old=0.0; // will not be used
   X_av=0.0; // will not be used
   MFEM_VERIFY(!Serial(), "internal MFEM error");
}

ParNonlinearForm::ParNonlinearForm(ParFiniteElementSpace *pf, ParFiniteElementSpace *pmf)
   : NonlinearForm(pf,pmf), pGrad(Operator::Hypre_ParCSR)
{
   X.MakeRef(pf, NULL);
   Y.MakeRef(pf, NULL);
   mX.MakeRef(pmf, NULL);
   X_old.MakeRef(pf, NULL);
   X_av=0.0; // will not be used
   MFEM_VERIFY(!Serial(), "internal MFEM error");
}

ParNonlinearForm::ParNonlinearForm(ParFiniteElementSpace *pf, ParFiniteElementSpace *pmf, ParFiniteElementSpace *pf_old)
   : NonlinearForm(pf,pmf,pf_old), pGrad(Operator::Hypre_ParCSR)
{
   X.MakeRef(pf, NULL);
   Y.MakeRef(pf, NULL);
   mX.MakeRef(pmf, NULL);
   X_old.MakeRef(pf_old, NULL);
   X_av=0.0; // will not be used
   MFEM_VERIFY(!Serial(), "internal MFEM error");
}

ParNonlinearForm::ParNonlinearForm(ParFiniteElementSpace *pf,
								   ParFiniteElementSpace *pmf,
								   ParFiniteElementSpace *pf_old,
								   ParFiniteElementSpace *pf_av)
   : NonlinearForm(pf,pmf,pf_old,pf_av), pGrad(Operator::Hypre_ParCSR)
{
   X.MakeRef(pf, NULL);
   Y.MakeRef(pf, NULL);
   mX.MakeRef(pmf, NULL);
   X_old.MakeRef(pf_old, NULL);
   X_av.MakeRef(pf_av, NULL);
   MFEM_VERIFY(!Serial(), "internal MFEM error");
}

double ParNonlinearForm::GetParGridFunctionEnergy(const Vector &x) const
{
   double loc_energy, glob_energy;

   loc_energy = GetGridFunctionEnergy(x);

   if (fnfi.Size())
   {
      MFEM_ABORT("TODO: add energy contribution from shared faces");
   }

   MPI_Allreduce(&loc_energy, &glob_energy, 1, MPI_DOUBLE, MPI_SUM,
                 ParFESpace()->GetComm());

   return glob_energy;
}

void ParNonlinearForm::Mult(const Vector &x, Vector &y) const
{
   NonlinearForm::Mult(x, y); // x --(P)--> aux1 --(A_local)--> aux2

   if (fnfi.Size())
   {
      MFEM_VERIFY(!NonlinearForm::ext, "Not implemented (extensions + faces");
      // Terms over shared interior faces in parallel.
      ParFiniteElementSpace *pfes = ParFESpace();
      ParMesh *pmesh = pfes->GetParMesh();
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs1, vdofs2;
      Vector el_x, el_y;

      aux1.HostReadWrite();
      X.MakeRef(aux1, 0); // aux1 contains P.x
      X.ExchangeFaceNbrData();
      const int n_shared_faces = pmesh->GetNSharedFaces();
      for (int i = 0; i < n_shared_faces; i++)
      {
         tr = pmesh->GetSharedFaceTransformations(i, true);
         int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

         fe1 = pfes->GetFE(tr->Elem1No);
         fe2 = pfes->GetFaceNbrFE(Elem2NbrNo);

         pfes->GetElementVDofs(tr->Elem1No, vdofs1);
         pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);

         el_x.SetSize(vdofs1.Size() + vdofs2.Size());
         X.GetSubVector(vdofs1, el_x.GetData());
         X.FaceNbrData().GetSubVector(vdofs2, el_x.GetData() + vdofs1.Size());

         for (int k = 0; k < fnfi.Size(); k++)
         {
            fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
            aux2.AddElementVector(vdofs1, el_y.GetData());
         }
      }
   }

   P->MultTranspose(aux2, y);

   const int N = ess_tdof_list.Size();
   const auto idx = ess_tdof_list.Read();
   auto Y_RW = y.ReadWrite();
   mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { Y_RW[idx[i]] = 0.0; });
}

void ParNonlinearForm::Mult(const Vector &x, const Vector &mx, Vector &y) const
{
   NonlinearForm::Mult(x, mx, y); // x --(P)--> aux1 --(A_local)--> aux2

   if (fnfi.Size())
   {
      MFEM_VERIFY(!NonlinearForm::ext, "Not implemented (extensions + faces");
      // Terms over shared interior faces in parallel.
      ParFiniteElementSpace *pfes = ParFESpace();
      ParFiniteElementSpace *pmfes = ParMFESpace();
      ParMesh *pmesh = pfes->GetParMesh();
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs1, vdofs2, mvdofs1, mvdofs2;
      Vector el_x, el_y, el_mx;

      aux1.HostReadWrite();
      X.MakeRef(aux1, 0); // aux1 contains P.x
      X.ExchangeFaceNbrData();

      maux.HostReadWrite();
      mX.MakeRef(maux, 0); // maux contains mP.mx
      mX.ExchangeFaceNbrData();

      const int n_shared_faces = pmesh->GetNSharedFaces();
      for (int i = 0; i < n_shared_faces; i++)
      {
         tr = pmesh->GetSharedFaceTransformations(i, true);
         int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

         fe1 = pfes->GetFE(tr->Elem1No);
         fe2 = pfes->GetFaceNbrFE(Elem2NbrNo);

         pfes->GetElementVDofs(tr->Elem1No, vdofs1);
         pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);

         el_x.SetSize(vdofs1.Size() + vdofs2.Size());
         X.GetSubVector(vdofs1, el_x.GetData());
         X.FaceNbrData().GetSubVector(vdofs2, el_x.GetData() + vdofs1.Size());

         pmfes->GetElementVDofs(tr->Elem1No, mvdofs1);
         pmfes->GetFaceNbrElementVDofs(Elem2NbrNo, mvdofs2);

         el_mx.SetSize(mvdofs1.Size() + mvdofs2.Size());
         mX.GetSubVector(mvdofs1, el_mx.GetData());
         mX.FaceNbrData().GetSubVector(mvdofs2, el_mx.GetData() + mvdofs1.Size());

         for (int k = 0; k < fnfi.Size(); k++)
         {
            fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_mx, el_y);
            aux2.AddElementVector(vdofs1, el_y.GetData());
         }
      }
   }

   P->MultTranspose(aux2, y);

   const int N = ess_tdof_list.Size();
   const auto idx = ess_tdof_list.Read();
   auto Y_RW = y.ReadWrite();
   mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { Y_RW[idx[i]] = 0.0; });
}

void ParNonlinearForm::Mult(const Vector &x, const Vector &mx, const Vector &xold, Vector &y) const
{
   NonlinearForm::Mult(x, mx, xold, y); // x --(P)--> aux1 --(A_local)--> aux2

   if (fnfi.Size())
   {
      MFEM_VERIFY(!NonlinearForm::ext, "Not implemented (extensions + faces");
      // Terms over shared interior faces in parallel.
      ParFiniteElementSpace *pfes = ParFESpace();
      ParFiniteElementSpace *pmfes = ParMFESpace();
      ParFiniteElementSpace *pfes_old = ParFESpace_old();

      ParMesh *pmesh = pfes->GetParMesh();
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs1, vdofs2, mvdofs1, mvdofs2, vdofs1_old, vdofs2_old;
      Vector el_x, el_y, el_mx, el_xold;

      aux1.HostReadWrite();
      X.MakeRef(aux1, 0); // aux1 contains P.x
      X.ExchangeFaceNbrData();

      maux.HostReadWrite();
      mX.MakeRef(maux, 0); // maux contains mP.mx
      mX.ExchangeFaceNbrData();

      aux_old.HostReadWrite();
      X_old.MakeRef(aux_old, 0); // aux_old contains P_old.xold
      X_old.ExchangeFaceNbrData();

      const int n_shared_faces = pmesh->GetNSharedFaces();
      for (int i = 0; i < n_shared_faces; i++)
      {
         tr = pmesh->GetSharedFaceTransformations(i, true);
         int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

         fe1 = pfes->GetFE(tr->Elem1No);
         fe2 = pfes->GetFaceNbrFE(Elem2NbrNo);

         pfes->GetElementVDofs(tr->Elem1No, vdofs1);
         pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);

         el_x.SetSize(vdofs1.Size() + vdofs2.Size());
         X.GetSubVector(vdofs1, el_x.GetData());
         X.FaceNbrData().GetSubVector(vdofs2, el_x.GetData() + vdofs1.Size());

         pmfes->GetElementVDofs(tr->Elem1No, mvdofs1);
         pmfes->GetFaceNbrElementVDofs(Elem2NbrNo, mvdofs2);

         el_mx.SetSize(mvdofs1.Size() + mvdofs2.Size());
         mX.GetSubVector(mvdofs1, el_mx.GetData());
         mX.FaceNbrData().GetSubVector(mvdofs2, el_mx.GetData() + mvdofs1.Size());

         pfes_old->GetElementVDofs(tr->Elem1No, vdofs1_old);
         pfes_old->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2_old);

         el_xold.SetSize(vdofs1_old.Size() + vdofs2_old.Size());
         X_old.GetSubVector(vdofs1_old, el_xold.GetData());
         X_old.FaceNbrData().GetSubVector(vdofs2_old, el_xold.GetData() + vdofs1_old.Size());

         for (int k = 0; k < fnfi.Size(); k++)
         {
            fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_mx, el_xold, el_y);
            aux2.AddElementVector(vdofs1, el_y.GetData());
         }
      }
   }

   P->MultTranspose(aux2, y);

   const int N = ess_tdof_list.Size();
   const auto idx = ess_tdof_list.Read();
   auto Y_RW = y.ReadWrite();
   mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { Y_RW[idx[i]] = 0.0; });
}

void ParNonlinearForm::Mult(const Vector &x, const Vector &mx, const Vector &xold, const Vector &xav, Vector &y) const
{
   NonlinearForm::Mult(x, mx, xold, xav, y); // x --(P)--> aux1 --(A_local)--> aux2

   if (fnfi.Size())
   {
      MFEM_VERIFY(!NonlinearForm::ext, "Not implemented (extensions + faces");
      // Terms over shared interior faces in parallel.
      ParFiniteElementSpace *pfes = ParFESpace();
      ParFiniteElementSpace *pmfes = ParMFESpace();
      ParFiniteElementSpace *pfes_old = ParFESpace_old();
      ParFiniteElementSpace *pfes_av = ParFESpace_av();

      ParMesh *pmesh = pfes->GetParMesh();
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs1, vdofs2, mvdofs1, mvdofs2, vdofs1_old, vdofs2_old, vdofs1_av, vdofs2_av;
      Vector el_x, el_y, el_mx, el_xold, el_av;

      aux1.HostReadWrite();
      X.MakeRef(aux1, 0); // aux1 contains P.x
      X.ExchangeFaceNbrData();

      maux.HostReadWrite();
      mX.MakeRef(maux, 0); // maux contains mP.mx
      mX.ExchangeFaceNbrData();

      aux_old.HostReadWrite();
      X_old.MakeRef(aux_old, 0); // aux_old contains P_old.xold
      X_old.ExchangeFaceNbrData();

      aux_av.HostReadWrite();
      X_av.MakeRef(aux_av, 0); // aux_av contains P_av.xav
      X_av.ExchangeFaceNbrData();

      const int n_shared_faces = pmesh->GetNSharedFaces();
      for (int i = 0; i < n_shared_faces; i++)
      {
         tr = pmesh->GetSharedFaceTransformations(i, true);
         int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

         fe1 = pfes->GetFE(tr->Elem1No);
         fe2 = pfes->GetFaceNbrFE(Elem2NbrNo);

         pfes->GetElementVDofs(tr->Elem1No, vdofs1);
         pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);

         el_x.SetSize(vdofs1.Size() + vdofs2.Size());
         X.GetSubVector(vdofs1, el_x.GetData());
         X.FaceNbrData().GetSubVector(vdofs2, el_x.GetData() + vdofs1.Size());

         pmfes->GetElementVDofs(tr->Elem1No, mvdofs1);
         pmfes->GetFaceNbrElementVDofs(Elem2NbrNo, mvdofs2);

         el_mx.SetSize(mvdofs1.Size() + mvdofs2.Size());
         mX.GetSubVector(mvdofs1, el_mx.GetData());
         mX.FaceNbrData().GetSubVector(mvdofs2, el_mx.GetData() + mvdofs1.Size());

         pfes_old->GetElementVDofs(tr->Elem1No, vdofs1_old);
         pfes_old->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2_old);

         el_xold.SetSize(vdofs1_old.Size() + vdofs2_old.Size());
         X_old.GetSubVector(vdofs1_old, el_xold.GetData());
         X_old.FaceNbrData().GetSubVector(vdofs2_old, el_xold.GetData() + vdofs1_old.Size());

         pfes_av->GetElementVDofs(tr->Elem1No, vdofs1_av);
         pfes_av->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2_av);

         el_av.SetSize(vdofs1_av.Size() + vdofs2_av.Size());
         X_av.GetSubVector(vdofs1_av, el_av.GetData());
         X_av.FaceNbrData().GetSubVector(vdofs2_av, el_av.GetData() + vdofs1_av.Size());

         for (int k = 0; k < fnfi.Size(); k++)
         {
            fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_mx, el_xold, el_av, el_y);
            aux2.AddElementVector(vdofs1, el_y.GetData());
         }
      }
   }

   P->MultTranspose(aux2, y);

   const int N = ess_tdof_list.Size();
   const auto idx = ess_tdof_list.Read();
   auto Y_RW = y.ReadWrite();
   mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { Y_RW[idx[i]] = 0.0; });
}

const SparseMatrix &ParNonlinearForm::GetLocalGradient(const Vector &x) const
{
   MFEM_VERIFY(NonlinearForm::ext == nullptr,
               "this method is not supported yet with partial assembly");

   NonlinearForm::GetGradient(x); // (re)assemble Grad, no b.c.

   return *Grad;
}

Operator &ParNonlinearForm::GetGradient(const Vector &x) const
{
   if (NonlinearForm::ext) { return NonlinearForm::GetGradient(x); }

   ParFiniteElementSpace *pfes = ParFESpace();

   pGrad.Clear();

   NonlinearForm::GetGradient(x); // (re)assemble Grad, no b.c.

   OperatorHandle dA(pGrad.Type()), Ph(pGrad.Type());

   if (fnfi.Size() == 0)
   {
      dA.MakeSquareBlockDiag(pfes->GetComm(), pfes->GlobalVSize(),
                             pfes->GetDofOffsets(), Grad);
   }
   else
   {
      MFEM_ABORT("TODO: assemble contributions from shared face terms");
   }

   // RAP the local gradient dA.
   // TODO - construct Dof_TrueDof_Matrix directly in the pGrad format
   Ph.ConvertFrom(pfes->Dof_TrueDof_Matrix());
   pGrad.MakePtAP(dA, Ph);

   // Impose b.c. on pGrad
   OperatorHandle pGrad_e;
   pGrad_e.EliminateRowsCols(pGrad, ess_tdof_list);

   return *pGrad.Ptr();
}

void ParNonlinearForm::Update()
{
   Y.MakeRef(ParFESpace(), NULL);
   X.MakeRef(ParFESpace(), NULL);
   pGrad.Clear();
   NonlinearForm::Update();
}


ParBlockNonlinearForm::ParBlockNonlinearForm(Array<ParFiniteElementSpace *> &pf)
   : BlockNonlinearForm()
{
   pBlockGrad = NULL;
   SetParSpaces(pf);
}

void ParBlockNonlinearForm::SetParSpaces(Array<ParFiniteElementSpace *> &pf)
{
   delete pBlockGrad;
   pBlockGrad = NULL;

   for (int s1=0; s1<fes.Size(); ++s1)
   {
      for (int s2=0; s2<fes.Size(); ++s2)
      {
         delete phBlockGrad(s1,s2);
      }
   }

   Array<FiniteElementSpace *> serialSpaces(pf.Size());

   for (int s=0; s<pf.Size(); s++)
   {
      serialSpaces[s] = (FiniteElementSpace *) pf[s];
   }

   SetSpaces(serialSpaces);

   phBlockGrad.SetSize(fes.Size(), fes.Size());

   for (int s1=0; s1<fes.Size(); ++s1)
   {
      for (int s2=0; s2<fes.Size(); ++s2)
      {
         phBlockGrad(s1,s2) = new OperatorHandle(Operator::Hypre_ParCSR);
      }
   }
}

ParFiniteElementSpace * ParBlockNonlinearForm::ParFESpace(int k)
{
   return (ParFiniteElementSpace *)fes[k];
}

const ParFiniteElementSpace *ParBlockNonlinearForm::ParFESpace(int k) const
{
   return (const ParFiniteElementSpace *)fes[k];
}

// Here, rhs is a true dof vector
void ParBlockNonlinearForm::SetEssentialBC(const
                                           Array<Array<int> *>&bdr_attr_is_ess,
                                           Array<Vector *> &rhs)
{
   Array<Vector *> nullarray(fes.Size());
   nullarray = NULL;

   BlockNonlinearForm::SetEssentialBC(bdr_attr_is_ess, nullarray);

   for (int s = 0; s < fes.Size(); ++s)
   {
      if (rhs[s])
      {
         rhs[s]->SetSubVector(*ess_tdofs[s], 0.0);
      }
   }
}

double ParBlockNonlinearForm::GetEnergy(const Vector &x) const
{
   // xs_true is not modified, so const_cast is okay
   xs_true.Update(const_cast<Vector &>(x), block_trueOffsets);
   xs.Update(block_offsets);

   for (int s = 0; s < fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->Mult(xs_true.GetBlock(s), xs.GetBlock(s));
   }

   double enloc = BlockNonlinearForm::GetEnergyBlocked(xs);
   double englo = 0.0;

   MPI_Allreduce(&enloc, &englo, 1, MPI_DOUBLE, MPI_SUM,
                 ParFESpace(0)->GetComm());

   return englo;
}

void ParBlockNonlinearForm::Mult(const Vector &x, Vector &y) const
{
   // xs_true is not modified, so const_cast is okay
   xs_true.Update(const_cast<Vector &>(x), block_trueOffsets);
   ys_true.Update(y, block_trueOffsets);
   xs.Update(block_offsets);
   ys.Update(block_offsets);

   for (int s=0; s<fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->Mult(
         xs_true.GetBlock(s), xs.GetBlock(s));
   }

   BlockNonlinearForm::MultBlocked(xs, ys);

   if (fnfi.Size() > 0)
   {
      MFEM_ABORT("TODO: assemble contributions from shared face terms");
   }

   for (int s=0; s<fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->MultTranspose(
         ys.GetBlock(s), ys_true.GetBlock(s));

      ys_true.GetBlock(s).SetSubVector(*ess_tdofs[s], 0.0);
   }

   ys_true.SyncFromBlocks();
   y.SyncMemory(ys_true);
}

/// Return the local gradient matrix for the given true-dof vector x
const BlockOperator & ParBlockNonlinearForm::GetLocalGradient(
   const Vector &x) const
{
   // xs_true is not modified, so const_cast is okay
   xs_true.Update(const_cast<Vector &>(x), block_trueOffsets);
   xs.Update(block_offsets);

   for (int s=0; s<fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->Mult(
         xs_true.GetBlock(s), xs.GetBlock(s));
   }

   // (re)assemble Grad without b.c. into 'Grads'
   BlockNonlinearForm::ComputeGradientBlocked(xs);

   delete BlockGrad;
   BlockGrad = new BlockOperator(block_offsets);

   for (int i = 0; i < fes.Size(); ++i)
   {
      for (int j = 0; j < fes.Size(); ++j)
      {
         BlockGrad->SetBlock(i, j, Grads(i, j));
      }
   }
   return *BlockGrad;
}

// Set the operator type id for the parallel gradient matrix/operator.
void ParBlockNonlinearForm::SetGradientType(Operator::Type tid)
{
   for (int s1=0; s1<fes.Size(); ++s1)
   {
      for (int s2=0; s2<fes.Size(); ++s2)
      {
         phBlockGrad(s1,s2)->SetType(tid);
      }
   }
}

BlockOperator & ParBlockNonlinearForm::GetGradient(const Vector &x) const
{
   if (pBlockGrad == NULL)
   {
      pBlockGrad = new BlockOperator(block_trueOffsets);
   }

   Array<const ParFiniteElementSpace *> pfes(fes.Size());

   for (int s1=0; s1<fes.Size(); ++s1)
   {
      pfes[s1] = ParFESpace(s1);

      for (int s2=0; s2<fes.Size(); ++s2)
      {
         phBlockGrad(s1,s2)->Clear();
      }
   }

   GetLocalGradient(x); // gradients are stored in 'Grads'

   if (fnfi.Size() > 0)
   {
      MFEM_ABORT("TODO: assemble contributions from shared face terms");
   }

   for (int s1=0; s1<fes.Size(); ++s1)
   {
      for (int s2=0; s2<fes.Size(); ++s2)
      {
         OperatorHandle dA(phBlockGrad(s1,s2)->Type()),
                        Ph(phBlockGrad(s1,s2)->Type()),
                        Rh(phBlockGrad(s1,s2)->Type());

         if (s1 == s2)
         {
            dA.MakeSquareBlockDiag(pfes[s1]->GetComm(), pfes[s1]->GlobalVSize(),
                                   pfes[s1]->GetDofOffsets(), Grads(s1,s1));
            Ph.ConvertFrom(pfes[s1]->Dof_TrueDof_Matrix());
            phBlockGrad(s1,s1)->MakePtAP(dA, Ph);

            OperatorHandle Ae;
            Ae.EliminateRowsCols(*phBlockGrad(s1,s1), *ess_tdofs[s1]);
         }
         else
         {
            dA.MakeRectangularBlockDiag(pfes[s1]->GetComm(),
                                        pfes[s1]->GlobalVSize(),
                                        pfes[s2]->GlobalVSize(),
                                        pfes[s1]->GetDofOffsets(),
                                        pfes[s2]->GetDofOffsets(),
                                        Grads(s1,s2));
            Rh.ConvertFrom(pfes[s1]->Dof_TrueDof_Matrix());
            Ph.ConvertFrom(pfes[s2]->Dof_TrueDof_Matrix());

            phBlockGrad(s1,s2)->MakeRAP(Rh, dA, Ph);

            phBlockGrad(s1,s2)->EliminateRows(*ess_tdofs[s1]);
            phBlockGrad(s1,s2)->EliminateCols(*ess_tdofs[s2]);
         }

         pBlockGrad->SetBlock(s1, s2, phBlockGrad(s1,s2)->Ptr());
      }
   }

   return *pBlockGrad;
}

ParBlockNonlinearForm::~ParBlockNonlinearForm()
{
   delete pBlockGrad;
   for (int s1=0; s1<fes.Size(); ++s1)
   {
      for (int s2=0; s2<fes.Size(); ++s2)
      {
         delete phBlockGrad(s1,s2);
      }
   }
}

}

#endif
