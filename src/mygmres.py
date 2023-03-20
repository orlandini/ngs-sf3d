from ngsolve.la import EigenValues_Preconditioner
from ngsolve import Projector, Norm, TimeFunction, BaseMatrix, Preconditioner, InnerProduct, \
    Norm, sqrt, Vector, Matrix, BaseVector, BlockVector, BitArray
from typing import Optional, Callable, Union
import logging
from netgen.libngpy._meshing import _PushStatus, _GetStatus, _SetThreadPercentage
from math import log
import os

from ngsolve.krylovspace import LinearSolver


class MyGMResSolver(LinearSolver):
    """Preconditioned GMRes solver. Minimizes the preconditioned residuum pre * (b-A*x)

Parameters
----------

innerproduct : Callable[[BaseVector, BaseVector], Union[float, complex]] = None
  Innerproduct to be used in iteration, all orthogonalizations/norms are computed with respect to that inner product.

restart : int = None
  If given, GMRes is restarted with the current solution x every 'restart' steps.
"""
    name = "MyGMRes"

    def __init__(self, *args,
                 innerproduct: Optional[Callable[[BaseVector, BaseVector],
                                                 Union[float, complex]]] = None,
                 restart: Optional[int] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if innerproduct is not None:
            self.innerproduct = innerproduct
            self.norm = lambda x: sqrt(innerproduct(x, x).real)
            self.restart = restart
        else:
            self.innerproduct = lambda x, y: y.InnerProduct(x, conjugate=True)
            self.norm = Norm
            self.restart = restart

    def _SolveImpl(self, rhs: BaseVector, sol: BaseVector):
        is_complex = rhs.is_complex
        A, pre, innerproduct, norm = self.mat, self.pre, self.innerproduct, self.norm
        n = len(rhs)
        m = self.maxiter
        sn = Vector(m, is_complex)
        cs = Vector(m, is_complex)
        sn[:] = 0
        cs[:] = 0
        if self.callback_sol is not None:
            sol_start = sol.CreateVector()
            sol_start.data = sol
        r = rhs.CreateVector()
        tmp = rhs.CreateVector()
        tmp.data = rhs - A * sol
        r.data = pre * tmp
        Q = []
        H = []
        Q.append(rhs.CreateVector())
        r_norm = norm(r)
        if self.CheckResidual(abs(r_norm)):
            return sol
        Q[0].data = 1./r_norm * r
        beta = Vector(m+1, is_complex)
        beta[:] = 0
        beta[0] = r_norm

        def arnoldi(A, Q, k):
            q = rhs.CreateVector()
            tmp.data = A * Q[k]
            q.data = pre * tmp
            h = Vector(m+1, is_complex)
            h[:] = 0
            for i in range(k+1):
                h[i] = innerproduct(Q[i], q)
                q.data += (-1) * h[i] * Q[i]
            h[k+1] = norm(q)
            if abs(h[k+1]) < 1e-12:
                return h, None
            q *= 1./h[k+1].real
            return h, q

        def givens_rotation(v1, v2):
            if v2 == 0:
                return 1, 0
            elif v1 == 0:
                return 0, v2/abs(v2)
            else:
                t = sqrt(v1.conjugate()*v1+v2*v2)
                cs = v1/t
                sn = v2/t
                return cs, sn

        def apply_givens_rotation(h, cs, sn, k):
            for i in range(k):
                temp = cs[i].conjugate() * h[i] + sn[i].conjugate() * h[i+1]
                h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1]
                h[i] = temp
            cs[k], sn[k] = givens_rotation(h[k], h[k+1])
            h[k] = cs[k] * h[k] + sn[k] * h[k+1]
            h[k+1] = 0

        def calcSolution(k):
            # if callback_sol is set we need to recompute solution in every step
            if self.callback_sol is not None:
                sol.data = sol_start
            mat = Matrix(k+1, k+1, is_complex)
            for i in range(k+1):
                mat[:, i] = H[i][:k+1]
            rs = Vector(k+1, is_complex)
            rs[:] = beta[:k+1]
            y = mat.I * rs
            for i in range(k+1):
                sol.data += y[i] * Q[i]

        for k in range(m):
            h, q = arnoldi(A, Q, k)
            H.append(h)
            if q is None:
                break
            Q.append(q)
            apply_givens_rotation(h, cs, sn, k)
            beta[k+1] = -sn[k].conjugate() * beta[k]
            beta[k] = cs[k] * beta[k]
            print("c {} s {} beta {}".format(cs[k], sn[k], beta[k+1]))
            error = abs(beta[k+1])
            if self.callback_sol is not None:
                calcSolution(k)
            if self.CheckResidual(error):
                break
            if self.restart is not None and (
                    k + 1 == self.restart and not (self.restart == self.maxiter)):
                calcSolution(k)
                del Q
                restarted_solver = GMResSolver(mat=self.mat,
                                               pre=self.pre,
                                               tol=0,
                                               atol=self._final_residual,
                                               callback=self.callback,
                                               callback_sol=self.callback_sol,
                                               maxiter=self.maxiter,
                                               restart=self.restart,
                                               printrates=self.printrates)
                restarted_solver.iterations = self.iterations
                sol = restarted_solver.Solve(rhs=rhs, sol=sol, initialize=False)
                self.residuals += restarted_solver.residuals
                self.iterations = restarted_solver.iterations
                return sol
        calcSolution(k)
        return sol


def MyGMRes(
        A, b, pre=None, freedofs=None, x=None, maxsteps=100, tol=None,
        innerproduct=None, callback=None, restart=None, startiteration=0,
        printrates=True, reltol=None):
    """Restarting preconditioned gmres solver for A*x=b. Minimizes the preconditioned residuum pre*(b-A*x).

Parameters
----------

A : BaseMatrix
  The left hand side of the linear system.

b : BaseVector
  The right hand side of the linear system.

pre : BaseMatrix = None
  The preconditioner for the system. If no preconditioner is given, the freedofs
  of the system must be given.

freedofs : BitArray = None
  Freedofs to solve on, only necessary if no preconditioner is given.

x : BaseVector = None
  Startvector, if given it will be modified in the routine and returned. Will be created
  if not given.

maxsteps : int = 100
  Maximum iteration steps.

tol : float = 1e-7

innerproduct : function = None
  Innerproduct to be used in iteration, all orthogonalizations/norms are computed with
  respect to that inner product.

callback : function = None
  If given, this function is called with the solution vector x in each step. Only for debugging

restart : int = None
  If given, gmres is restarted with the current solution x every 'restart' steps.

startiteration : int = 0
  Internal value to count total number of iterations in restarted setup, no user input required
  here.

printrates : bool = True
  Print norm of preconditioned residual in each step.
"""
    solver = MyGMResSolver(mat=A, pre=pre, freedofs=freedofs,
                           maxiter=maxsteps, tol=reltol, atol=tol,
                           innerproduct=innerproduct,
                           callback_sol=callback, restart=restart,
                           printrates=printrates)
    return solver.Solve(rhs=b, sol=x)
