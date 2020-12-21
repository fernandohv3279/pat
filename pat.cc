/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */



#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>
#include <fstream>
#include <iostream>
#include <deal.II/base/function_lib.h>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <deal.II/numerics/fe_field_function.h>

using namespace dealii;

#define deltaTime 0.00457646
#define stepsInTime 327

class bound : public Function<2>
{
	public:
		bound();
		virtual double value(const Point<2> &p,
			     const unsigned int component = 0) const override
  		{
    			return boundary_data.value(
      			Point<2>(deltaTime*stepsInTime-this->get_time(),std::atan2(p[1],p[0])*180/numbers::PI));

  		}
	private:
		const Functions::InterpolatedUniformGridData<2> boundary_data;
		static std::vector<double> get_data();
};

bound::bound()
	: boundary_data({{std::make_pair(deltaTime, deltaTime*stepsInTime),
			          std::make_pair(0, 160*2.25)}},
                                {{stepsInTime-1, 160}},
				Table<2, double>(stepsInTime, 161, get_data().begin()))
  {}

  std::vector<double> bound::get_data()
  {
    std::vector<double> data;
    boost::iostreams::filtering_istream in;
    in.push(boost::iostreams::file_source("detectors.dat"));
    double time, z;
    for (unsigned int line = 0; line < stepsInTime; ++line)
	{
		in >> time;
	for (unsigned int col=0; col<161; ++col)
    	  {
              try
          	{
            	in >> z;
            	data.push_back(z);
          	}
              catch (...)
	  	{
	    	AssertThrow(false,
	         	       ExcMessage("Could not read all data points "
                                   "from the file <detectors.dat>!"));
          	}
      	  }
	}
    return data;
  }



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++WAVE ecuation++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
namespace Step23
{
  using namespace dealii;



  template <int dim>
  class WaveEquation
  {
  public:
    WaveEquation();
    void run();

  private:
    void setup_system();
    void solve_u();
    void solve_v();
    void output_results() const;

    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> matrix_u;
    SparseMatrix<double> matrix_v;

    Vector<double> solution_u, solution_v;
    Vector<double> old_solution_u, old_solution_v;
    Vector<double> system_rhs;

    double       time_step;
    double       time;
    unsigned int timestep_number;
    const double theta;
  };





  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> & /*p*/,
                         const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));
      return 0;
    }
  };






  template <int dim>
  class BoundaryValuesV : public Function<dim>
  {
  public:
    BoundaryValuesV(bound& boundaryF)
    {
	    bdry=&boundaryF;
    }
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      bdry->set_time(this->get_time());
      if(this->get_time()<deltaTime+0.00001)
	{
        double ft= bdry->value(p);
      	bdry->set_time(bdry->get_time()+deltaTime);
	double fT=bdry->value(p);
      	bdry->set_time(this->get_time());
	return (fT-ft)/deltaTime;
        }
      else if (this->get_time()>deltaTime*stepsInTime-0.00001)
	{
        double fT= bdry->value(p);
      	bdry->set_time(bdry->get_time()-deltaTime);
	double ft=bdry->value(p);
      	bdry->set_time(this->get_time());
	return (fT-ft)/deltaTime;
	}
      else
	{
        double f= bdry->value(p);
      	bdry->set_time(bdry->get_time()+deltaTime);
	double fT=bdry->value(p);
      	bdry->set_time(this->get_time());
      	bdry->set_time(bdry->get_time()-deltaTime);
	double ft=bdry->value(p);
      	bdry->set_time(this->get_time());
	return (fT-2*f+ft)/(deltaTime*deltaTime);
	}
    }
  private:
    bound* bdry;
  };



  // @sect3{Implementation of the <code>WaveEquation</code> class}

  template <int dim>
  WaveEquation<dim>::WaveEquation()
    : fe(2)
    , dof_handler(triangulation)
    //, time_step(1. / 64)
    , time_step(0.00457646)
    , time(time_step)
    , timestep_number(1)
    , theta(0.5)
  {}


  // @sect4{WaveEquation::setup_system}

  template <int dim>
  void WaveEquation<dim>::setup_system()
  {
    const Point<dim> center;
    GridGenerator::hyper_ball(triangulation, center, 0.5);
    triangulation.refine_global(5);

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl;

    dof_handler.distribute_dofs(fe);

    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    matrix_u.reinit(sparsity_pattern);
    matrix_v.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler,
                                         QGauss<dim>(fe.degree + 1),
                                         laplace_matrix);

    solution_u.reinit(dof_handler.n_dofs());
    solution_v.reinit(dof_handler.n_dofs());
    old_solution_u.reinit(dof_handler.n_dofs());
    old_solution_v.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.close();
  }



  // @sect4{WaveEquation::solve_u and WaveEquation::solve_v}

  template <int dim>
  void WaveEquation<dim>::solve_u()
  {
    SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    cg.solve(matrix_u, solution_u, system_rhs, PreconditionIdentity());

    std::cout << "   u-equation: " << solver_control.last_step()
              << " CG iterations." << std::endl;
  }



  template <int dim>
  void WaveEquation<dim>::solve_v()
  {
    SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    cg.solve(matrix_v, solution_v, system_rhs, PreconditionIdentity());

    std::cout << "   v-equation: " << solver_control.last_step()
              << " CG iterations." << std::endl;
  }



  // @sect4{WaveEquation::output_results}

  template <int dim>
  void WaveEquation<dim>::output_results() const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution_u, "U");
    data_out.add_data_vector(solution_v, "V");

    data_out.build_patches();

    const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";
    // Like step-15, since we write output at every time step (and the system
    // we have to solve is relatively easy), we instruct DataOut to use the
    // zlib compression algorithm that is optimized for speed instead of disk
    // usage since otherwise plotting the output becomes a bottleneck:
    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level =
      DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);
    std::ofstream output(filename);
    data_out.write_vtu(output);
  }



  // @sect4{WaveEquation::run}

  // The following is really the only interesting function of the program. It
  // contains the loop over all time steps, but before we get to that we have
  // to set up the grid, DoFHandler, and matrices. In addition, we have to
  // somehow get started with initial values. To this end, we use the
  // VectorTools::project function that takes an object that describes a
  // continuous function and computes the $L^2$ projection of this function
  // onto the finite element space described by the DoFHandler object. Can't
  // be any simpler than that:
  template <int dim>
  void WaveEquation<dim>::run()
  {
    setup_system();
  //////////////////////////////InitialValues/////////////////////////////////
  boost::iostreams::filtering_istream inPoisson;
  inPoisson.push(boost::iostreams::file_source("sol.dat"));
  double solPointPoisson;
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
	{
		inPoisson >> solPointPoisson;
		old_solution_u=solPointPoisson;
                old_solution_v=0;
	}
  //////////////////////////////InitialValues/////////////////////////////////
    // The next thing is to loop over all the time steps until we reach the
    // end time ($T=5$ in this case). In each time step, we first have to
    // solve for $U^n$, using the equation $(M^n + k^2\theta^2 A^n)U^n =$
    // $(M^{n,n-1} - k^2\theta(1-\theta) A^{n,n-1})U^{n-1} + kM^{n,n-1}V^{n-1}
    // +$ $k\theta \left[k \theta F^n + k(1-\theta) F^{n-1} \right]$. Note
    // that we use the same mesh for all time steps, so that $M^n=M^{n,n-1}=M$
    // and $A^n=A^{n,n-1}=A$. What we therefore have to do first is to add up
    // $MU^{n-1} - k^2\theta(1-\theta) AU^{n-1} + kMV^{n-1}$ and the forcing
    // terms, and put the result into the <code>system_rhs</code> vector. (For
    // these additions, we need a temporary vector that we declare before the
    // loop to avoid repeated memory allocations in each time step.)
    //
    Vector<double> tmp(solution_u.size());
    Vector<double> forcing_terms(solution_u.size());

    for (; time <= 1.5; time += time_step, ++timestep_number)
      {
        std::cout << "Time step " << timestep_number << " at t=" << time
                  << std::endl;

        mass_matrix.vmult(system_rhs, old_solution_u);

        mass_matrix.vmult(tmp, old_solution_v);
        system_rhs.add(time_step, tmp);

        laplace_matrix.vmult(tmp, old_solution_u);
        system_rhs.add(-theta * (1 - theta) * time_step * time_step, tmp);

        RightHandSide<dim> rhs_function;
        rhs_function.set_time(time);
        VectorTools::create_right_hand_side(dof_handler,
                                            QGauss<dim>(fe.degree + 1),
                                            rhs_function,
                                            tmp);
        forcing_terms = tmp;
        forcing_terms *= theta * time_step;

        rhs_function.set_time(time - time_step);
        VectorTools::create_right_hand_side(dof_handler,
                                            QGauss<dim>(fe.degree + 1),
                                            rhs_function,
                                            tmp);

        forcing_terms.add((1 - theta) * time_step, tmp);

        system_rhs.add(theta * time_step, forcing_terms);

        // After so constructing the right hand side vector of the first
        // equation, all we have to do is apply the correct boundary
        // values. As for the right hand side, this is a space-time function
        // evaluated at a particular time, which we interpolate at boundary
        // nodes and then use the result to apply boundary values as we
        // usually do. The result is then handed off to the solve_u()
        // function:
        {
          //BoundaryValuesU<dim> boundary_values_u_function;
          //bound boundary_values_u_function;
          bound boundary_values_u_function;
          boundary_values_u_function.set_time(time);

          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   0,
                                                   boundary_values_u_function,
                                                   boundary_values);

          // The matrix for solve_u() is the same in every time steps, so one
          // could think that it is enough to do this only once at the
          // beginning of the simulation. However, since we need to apply
          // boundary values to the linear system (which eliminate some matrix
          // rows and columns and give contributions to the right hand side),
          // we have to refill the matrix in every time steps before we
          // actually apply boundary data. The actual content is very simple:
          // it is the sum of the mass matrix and a weighted Laplace matrix:
          matrix_u.copy_from(mass_matrix);
          matrix_u.add(theta * theta * time_step * time_step, laplace_matrix);
          MatrixTools::apply_boundary_values(boundary_values,
                                             matrix_u,
                                             solution_u,
                                             system_rhs);
        }
        solve_u();


        // The second step, i.e. solving for $V^n$, works similarly, except
        // that this time the matrix on the left is the mass matrix (which we
        // copy again in order to be able to apply boundary conditions, and
        // the right hand side is $MV^{n-1} - k\left[ \theta A U^n +
        // (1-\theta) AU^{n-1}\right]$ plus forcing terms. Boundary values
        // are applied in the same way as before, except that now we have to
        // use the BoundaryValuesV class:
        laplace_matrix.vmult(system_rhs, solution_u);
        system_rhs *= -theta * time_step;

        mass_matrix.vmult(tmp, old_solution_v);
        system_rhs += tmp;

        laplace_matrix.vmult(tmp, old_solution_u);
        system_rhs.add(-time_step * (1 - theta), tmp);

        system_rhs += forcing_terms;

        {
          bound boundary_values_u_function2;
          BoundaryValuesV<dim> boundary_values_v_function(boundary_values_u_function2);
          boundary_values_v_function.set_time(time);

          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   0,
                                                   boundary_values_v_function,
                                                   boundary_values);
          matrix_v.copy_from(mass_matrix);
          MatrixTools::apply_boundary_values(boundary_values,
                                             matrix_v,
                                             solution_v,
                                             system_rhs);
        }
        solve_v();

        // Finally, after both solution components have been computed, we
        // output the result, compute the energy in the solution, and go on to
        // the next time step after shifting the present solution into the
        // vectors that hold the solution at the previous time step. Note the
        // function SparseMatrix::matrix_norm_square that can compute
        // $\left<V^n,MV^n\right>$ and $\left<U^n,AU^n\right>$ in one step,
        // saving us the expense of a temporary vector and several lines of
        // code:
        output_results();

        std::cout << "   Total energy: "
                  << (mass_matrix.matrix_norm_square(solution_v) +
                      laplace_matrix.matrix_norm_square(solution_u)) /
                       2
                  << std::endl;

        old_solution_u = solution_u;
        old_solution_v = solution_v;
      }
  }
} // namespace Step23



int main()
{
  try
    {
      using namespace Step23;

      WaveEquation<2> wave_equation_solver;
      wave_equation_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
