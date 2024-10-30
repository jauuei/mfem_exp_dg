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

#ifndef MFEM_NONLINEARFORM
#define MFEM_NONLINEARFORM

#include "../config/config.hpp"
#include "nonlininteg.hpp"
#include "nonlinearform_ext.hpp"
#include "bilinearform.hpp"
#include "gridfunc.hpp"

namespace mfem
{

class NonlinearForm : public Operator
{
protected:
   /// The assembly level.
   AssemblyLevel assembly;

   /// Extension for supporting different AssemblyLevel%s
   /** For nonlinear operators, the "matrix" assembly levels usually do not make
       sense, so only PARTIAL and NONE (matrix-free) are supported. */
   NonlinearFormExtension *ext; // owned

   /// FE space on which the form lives.
   FiniteElementSpace *fes; // not owned

   /// Mixed FE space on which the form lives.
   FiniteElementSpace *mfes; // not owned

   /// FE space on which the form lives. The old (previous solution) is defined on this space.
   //  It is used in linear evaluation in exponential time integrators.
   //  In practice, fes_old is actually equal to fes.
   FiniteElementSpace *fes_old; // not owned

   // FE space used for defining artificial viscosity/resistivity
   FiniteElementSpace *fes_av; // not owned

   /// Set of Domain Integrators to be assembled (added).
   Array<NonlinearFormIntegrator*> dnfi; // owned

   /// Set of interior face Integrators to be assembled (added).
   Array<NonlinearFormIntegrator*> fnfi; // owned

   /// Set of boundary face Integrators to be assembled (added).
   Array<NonlinearFormIntegrator*> bfnfi; // owned
   Array<Array<int>*>              bfnfi_marker; // not owned

   mutable SparseMatrix *Grad, *cGrad; // owned
   /// Gradient Operator when not assembled as a matrix.
   mutable OperatorHandle hGrad; // has internal ownership flag

   /// A list of all essential true dofs
   Array<int> ess_tdof_list;

   /// Counter for updates propagated from the FiniteElementSpace.
   long sequence;

   /// Auxiliary Vector%s
   mutable Vector aux1, aux2, maux, aux_old, aux_av;

   /// Pointer to the prolongation matrix of fes, may be NULL.
   const Operator *P; // not owned
   /// Pointer to the prolongation matrix of mfes, may be NULL.
   const Operator *mP; // not owned
   /// Pointer to the prolongation matrix of fes_old, may be NULL.
   const Operator *P_old; // not owned
   /// Pointer to the prolongation matrix of fes_av, may be NULL.
   const Operator *P_av; // not owned
   /// The result of dynamic-casting P to SparseMatrix pointer.
   const SparseMatrix *cP; // not owned

   bool Serial() const { return (!P || cP); }
   const Vector &Prolongate(const Vector &x) const;

   // The function same as Prolongate but acts on the vector defined on the mfes.
   const Vector &mProlongate(const Vector &x) const;

   // The function same as Prolongate but acts on the vector defined on the fes_old.
   const Vector &Prolongate_old(const Vector &x) const;

   // The function same as Prolongate but acts on the vector defined on the fes_av.
   const Vector &Prolongate_av(const Vector &x) const;
public:
   /// Construct a NonlinearForm on the given FiniteElementSpace, @a f.
   /** As an Operator, the NonlinearForm has input and output size equal to the
       number of true degrees of freedom, i.e. f->GetTrueVSize(). */
   NonlinearForm(FiniteElementSpace *f)
      : Operator(f->GetTrueVSize()), assembly(AssemblyLevel::LEGACY),
        ext(NULL), fes(f), Grad(NULL), cGrad(NULL),
        sequence(f->GetSequence()), P(f->GetProlongationMatrix()),
        cP(dynamic_cast<const SparseMatrix*>(P)), mfes(nullptr), mP(nullptr),
		fes_old(nullptr), P_old(nullptr),
		fes_av(nullptr),P_av(nullptr)

   { }

   /// Construct a (mixed) NonlinearForm on the given FiniteElementSpace, @a f and @a mf.
   /** As an Operator, the NonlinearForm has input and output size equal to the
       number of true degrees of freedom, i.e. f->GetTrueVSize(). */
   NonlinearForm(FiniteElementSpace *f, FiniteElementSpace *mf)
      : Operator(f->GetTrueVSize()), assembly(AssemblyLevel::LEGACY),
        ext(NULL), fes(f), Grad(NULL), cGrad(NULL),
        sequence(f->GetSequence()), P(f->GetProlongationMatrix()),
        cP(dynamic_cast<const SparseMatrix*>(P)), mfes(mf), mP(mf->GetProlongationMatrix()),
		fes_old(nullptr), P_old(nullptr),
		fes_av(nullptr),P_av(nullptr)
   { }

   /// Construct a (mixed) NonlinearForm on the given FiniteElementSpace, @a f, @a mf and @a f_old.
   /** As an Operator, the NonlinearForm has input and output size equal to the
       number of true degrees of freedom, i.e. f->GetTrueVSize(). */
   NonlinearForm(FiniteElementSpace *f, FiniteElementSpace *mf, FiniteElementSpace *f_old)
      : Operator(f->GetTrueVSize()), assembly(AssemblyLevel::LEGACY),
        ext(NULL), fes(f), Grad(NULL), cGrad(NULL),
        sequence(f->GetSequence()), P(f->GetProlongationMatrix()),
        cP(dynamic_cast<const SparseMatrix*>(P)), mfes(mf), mP(mf->GetProlongationMatrix()),
		fes_old(f_old), P_old(f_old->GetProlongationMatrix()),
		fes_av(nullptr),P_av(nullptr)
   { }

   /// Construct a (mixed) NonlinearForm on the given FiniteElementSpace, @a f, @a mf, @a f_old and @a f_av.
   /** As an Operator, the NonlinearForm has input and output size equal to the
       number of true degrees of freedom, i.e. f->GetTrueVSize(). */
   NonlinearForm(FiniteElementSpace *f, FiniteElementSpace *mf, FiniteElementSpace *f_old, FiniteElementSpace *f_av)
      : Operator(f->GetTrueVSize()), assembly(AssemblyLevel::LEGACY),
        ext(NULL), fes(f), Grad(NULL), cGrad(NULL),
        sequence(f->GetSequence()), P(f->GetProlongationMatrix()),
        cP(dynamic_cast<const SparseMatrix*>(P)), mfes(mf), mP(mf->GetProlongationMatrix()),
		fes_old(f_old), P_old(f_old->GetProlongationMatrix()),
		fes_av(f_av),P_av(f_av->GetProlongationMatrix())
   { }

   /// Set the desired assembly level. The default is AssemblyLevel::LEGACY.
   /** For nonlinear operators, the "matrix" assembly levels usually do not make
       sense, so only LEGACY, NONE (matrix-free) and PARTIAL are supported.

       Currently, AssemblyLevel::LEGACY uses the standard nonlinear action
       methods like AssembleElementVector of the NonlinearFormIntegrator class
       which work only on CPU and do not utilize features such as fast
       tensor-product basis evaluations. In this mode, the gradient operator is
       constructed as a SparseMatrix (or, in parallel, format such as
       HypreParMatrix).

       When using AssemblyLevel::PARTIAL, the action is performed using methods
       like AddMultPA of the NonlinearFormIntegrator class which typically
       support both CPU and GPU backends and utilize features such as fast
       tensor-product basis evaluations. In this mode, the gradient operator
       also uses partial assembly with support for CPU and GPU backends.

       When using AssemblyLevel::NONE, the action is performed using methods
       like AddMultMF of the NonlinearFormIntegrator class which typically
       support both CPU and GPU backends and utilize features such as fast
       tensor-product basis evaluations. In this mode, the gradient operator
       is currently not supported.

       This method must be called before "assembly" with Setup(). */
   void SetAssemblyLevel(AssemblyLevel assembly_level);

   FiniteElementSpace *FESpace() { return fes; }
   const FiniteElementSpace *FESpace() const { return fes; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(NonlinearFormIntegrator *nlfi)
   { dnfi.Append(nlfi); }

   /// Access all integrators added with AddDomainIntegrator().
   Array<NonlinearFormIntegrator*> *GetDNFI() { return &dnfi; }
   const Array<NonlinearFormIntegrator*> *GetDNFI() const { return &dnfi; }

   /// Adds new Interior Face Integrator.
   void AddInteriorFaceIntegrator(NonlinearFormIntegrator *nlfi)
   { fnfi.Append(nlfi); }

   /** @brief Access all interior face integrators added with
       AddInteriorFaceIntegrator(). */
   const Array<NonlinearFormIntegrator*> &GetInteriorFaceIntegrators() const
   { return fnfi; }

   /// Adds new Boundary Face Integrator.
   void AddBdrFaceIntegrator(NonlinearFormIntegrator *nlfi)
   { bfnfi.Append(nlfi); bfnfi_marker.Append(NULL); }

   /** @brief Adds new Boundary Face Integrator, restricted to specific boundary
       attributes. */
   void AddBdrFaceIntegrator(NonlinearFormIntegrator *nfi,
                             Array<int> &bdr_marker)
   { bfnfi.Append(nfi); bfnfi_marker.Append(&bdr_marker); }

   /** @brief Access all boundary face integrators added with
       AddBdrFaceIntegrator(). */
   const Array<NonlinearFormIntegrator*> &GetBdrFaceIntegrators() const
   { return bfnfi; }

   /// Specify essential boundary conditions.
   /** This method calls FiniteElementSpace::GetEssentialTrueDofs() and stores
       the result internally for use by other methods. If the @a rhs pointer is
       not NULL, its essential true dofs will be set to zero. This makes it
       "compatible" with the output vectors from the Mult() method which also
       have zero entries at the essential true dofs. */
   void SetEssentialBC(const Array<int> &bdr_attr_is_ess, Vector *rhs = NULL);

   /// Specify essential boundary conditions.
   /** Use either SetEssentialBC() or SetEssentialTrueDofs() if possible. */
   void SetEssentialVDofs(const Array<int> &ess_vdofs_list);

   /// Specify essential boundary conditions.
   void SetEssentialTrueDofs(const Array<int> &ess_tdof_list_)
   { ess_tdof_list_.Copy(this->ess_tdof_list); }

   /// Return a (read-only) list of all essential true dofs.
   const Array<int> &GetEssentialTrueDofs() const { return ess_tdof_list; }

   /// Compute the energy corresponding to the state @a x.
   /** In general, @a x may have non-homogeneous essential boundary values.

       The state @a x must be a "GridFunction size" vector, i.e. its size must
       be fes->GetVSize(). */
   double GetGridFunctionEnergy(const Vector &x) const;

   /// Compute the energy corresponding to the state @a x.
   /** In general, @a x may have non-homogeneous essential boundary values.

       The state @a x must be a true-dof vector. */
   virtual double GetEnergy(const Vector &x) const
   { return GetGridFunctionEnergy(Prolongate(x)); }

   /// Evaluate the action of the NonlinearForm.
   /** The input essential dofs in @a x will, generally, be non-zero. However,
       the output essential dofs in @a y will always be set to zero.

       Both the input and the output vectors, @a x and @a y, must be true-dof
       vectors, i.e. their size must be fes->GetTrueVSize(). */
   virtual void Mult(const Vector &x, Vector &y) const;

   /// Evaluate the action of the NonlinearForm.
   /** The input essential dofs in @a x will, generally, be non-zero. However,
       the output essential dofs in @a y will always be set to zero.

       Both the input and the output vectors, @a x and @a y, must be true-dof
       vectors, i.e. their size must be fes->GetTrueVSize().

       The input @a mx should follow the same rule applied to @a x. The function
       is typically used for scenario where a mixed finite element space is applied.
       (i.e., both @a x and @a mx could contribute to computation of @a y) */
   virtual void Mult(const Vector &x, const Vector &mx,  Vector &y) const;

   /// Evaluate the action of the NonlinearForm.
   /** The input essential dofs in @a x will, generally, be non-zero. However,
       the output essential dofs in @a y will always be set to zero.

       Both the input and the output vectors, @a x and @a y, must be true-dof
       vectors, i.e. their size must be fes->GetTrueVSize().

       The input @a mx should follow the same rule applied to @a x. The function
       is typically used for scenario where a mixed finite element space is applied.
       (i.e., both @a x and @a mx could contribute to computation of @a y)

       The input @ x_old also follow the same rule. This function is used in the linear
       evaluation in the exponential time integrators.
        */
   virtual void Mult(const Vector &x, const Vector &mx, const Vector &x_old,  Vector &y) const;

   /// Evaluate the action of the NonlinearForm.
   /** The input essential dofs in @a x will, generally, be non-zero. However,
       the output essential dofs in @a y will always be set to zero.

       Both the input and the output vectors, @a x and @a y, must be true-dof
       vectors, i.e. their size must be fes->GetTrueVSize().

       The input @a mx should follow the same rule applied to @a x. The function
       is typically used for scenario where a mixed finite element space is applied.
       (i.e., both @a x and @a mx could contribute to computation of @a y)

       The input @ x_old also follow the same rule. This function is used in the linear
       evaluation in the exponential time integrators.

       The input @ x_av also follow the same rule. This function is used in the linear
       evaluation in the exponential time integrators along with artificial viscosity/resistivity.
        */
   virtual void Mult(const Vector &x, const Vector &mx, const Vector &x_old, const Vector &x_av,  Vector &y) const;

   /** @brief Compute the gradient Operator of the NonlinearForm corresponding
       to the state @a x. */
   /** Any previously specified essential boundary conditions will be
       automatically imposed on the gradient operator.

       The returned object is valid until the next call to this method or the
       destruction of this object.

       In general, @a x may have non-homogeneous essential boundary values.

       The state @a x must be a true-dof vector. */
   virtual Operator &GetGradient(const Vector &x) const;

   /// Update the NonlinearForm to propagate updates of the associated FE space.
   /** After calling this method, the essential boundary conditions need to be
       set again. */
   virtual void Update();

   /** @brief Setup the NonlinearForm: based on the current AssemblyLevel and
       the current mesh, optionally, precompute and store data that will be
       reused in subsequent call to Mult(). */
   /** Typically, this method has to be called before Mult() when using
       AssemblyLevel::PARTIAL, after calling Update(), or after modifying the
       mesh coordinates. */
   virtual void Setup();

   /// Get the finite element space prolongation matrix
   virtual const Operator *GetProlongation() const { return P; }
   /// Get the finite element space restriction matrix
   virtual const Operator *GetRestriction() const
   { return fes->GetRestrictionMatrix(); }

   /** @brief Destroy the NonlinearForm including the owned
       NonlinearFormIntegrator%s and gradient Operator. */
   virtual ~NonlinearForm();
};


/** @brief A class representing a general block nonlinear operator defined on
    the Cartesian product of multiple FiniteElementSpace%s. */
class BlockNonlinearForm : public Operator
{
protected:
   /// FE spaces on which the form lives.
   Array<FiniteElementSpace*> fes;

   /// Set of Domain Integrators to be assembled (added).
   Array<BlockNonlinearFormIntegrator*> dnfi;

   /// Set of interior face Integrators to be assembled (added).
   Array<BlockNonlinearFormIntegrator*> fnfi;

   /// Set of Boundary Face Integrators to be assembled (added).
   Array<BlockNonlinearFormIntegrator*> bfnfi;
   Array<Array<int>*>           bfnfi_marker;

   /** Auxiliary block-vectors for wrapping input and output vectors or holding
       GridFunction-like block-vector data (e.g. in parallel). */
   mutable BlockVector xs, ys;

   mutable Array2D<SparseMatrix*> Grads, cGrads;
   mutable BlockOperator *BlockGrad;

   // A list of the offsets
   Array<int> block_offsets;
   Array<int> block_trueOffsets;

   // Array of Arrays of tdofs for each space in 'fes'
   Array<Array<int> *> ess_tdofs;

   /// Array of pointers to the prolongation matrix of fes, may be NULL
   Array<const Operator *> P;

   /// Array of results of dynamic-casting P to SparseMatrix pointer
   Array<const SparseMatrix *> cP;

   /// Indicator if the Operator is part of a parallel run
   bool is_serial = true;

   /// Indicator if the Operator needs prolongation on assembly
   bool needs_prolongation = false;

   mutable BlockVector aux1, aux2;

   const BlockVector &Prolongate(const BlockVector &bx) const;

   /// Specialized version of GetEnergy() for BlockVectors
   double GetEnergyBlocked(const BlockVector &bx) const;

   /// Specialized version of Mult() for BlockVector%s
   /// Block L-Vector to Block L-Vector
   void MultBlocked(const BlockVector &bx, BlockVector &by) const;

   /// Specialized version of GetGradient() for BlockVector
   void ComputeGradientBlocked(const BlockVector &bx) const;

public:
   /// Construct an empty BlockNonlinearForm. Initialize with SetSpaces().
   BlockNonlinearForm();

   /// Construct a BlockNonlinearForm on the given set of FiniteElementSpace%s.
   BlockNonlinearForm(Array<FiniteElementSpace *> &f);

   /// Return the @a k-th FE space of the BlockNonlinearForm.
   FiniteElementSpace *FESpace(int k) { return fes[k]; }
   /// Return the @a k-th FE space of the BlockNonlinearForm (const version).
   const FiniteElementSpace *FESpace(int k) const { return fes[k]; }

   /// (Re)initialize the BlockNonlinearForm.
   /** After a call to SetSpaces(), the essential b.c. must be set again. */
   void SetSpaces(Array<FiniteElementSpace *> &f);

   /// Return the regular dof offsets.
   const Array<int> &GetBlockOffsets() const { return block_offsets; }
   /// Return the true-dof offsets.
   const Array<int> &GetBlockTrueOffsets() const { return block_trueOffsets; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(BlockNonlinearFormIntegrator *nlfi)
   { dnfi.Append(nlfi); }

   /// Adds new Interior Face Integrator.
   void AddInteriorFaceIntegrator(BlockNonlinearFormIntegrator *nlfi)
   { fnfi.Append(nlfi); }

   /// Adds new Boundary Face Integrator.
   void AddBdrFaceIntegrator(BlockNonlinearFormIntegrator *nlfi)
   { bfnfi.Append(nlfi); bfnfi_marker.Append(NULL); }

   /** @brief Adds new Boundary Face Integrator, restricted to specific boundary
       attributes. */
   void AddBdrFaceIntegrator(BlockNonlinearFormIntegrator *nlfi,
                             Array<int> &bdr_marker);

   virtual void SetEssentialBC(const Array<Array<int> *>&bdr_attr_is_ess,
                               Array<Vector *> &rhs);

   virtual double GetEnergy(const Vector &x) const;

   /// Method is only called in serial, the parallel version calls MultBlocked
   /// directly.
   virtual void Mult(const Vector &x, Vector &y) const;

   /// Method is only called in serial, the parallel version calls
   /// GetGradientBlocked directly.
   virtual Operator &GetGradient(const Vector &x) const;

   /// Destructor.
   virtual ~BlockNonlinearForm();
};


}

#endif
