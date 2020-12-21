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

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
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

class Step3
{
public:
  Step3();

  void run();


private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<2> triangulation;
  FE_Q<2>          fe;
  DoFHandler<2>    dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};


Step3::Step3()
  : fe(2)
  , dof_handler(triangulation)
{}



void Step3::make_grid()
{
  const Point<2> center;
  GridGenerator::hyper_ball(triangulation, center, 0.5);
  triangulation.refine_global(5);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}




void Step3::setup_system()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}



void Step3::assemble_system()
{
  QGauss<2> quadrature_formula(fe.degree + 1);
  FEValues<2> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      cell_matrix = 0;
      cell_rhs    = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
		      (fe_values.shape_grad(i, q_index) *
		       fe_values.shape_grad(j, q_index) *
		       fe_values.JxW(q_index));

          for (const unsigned int i : fe_values.dof_indices())
		  cell_rhs(i) += 0;
        }
      cell->get_dof_indices(local_dof_indices);

      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));

      for (const unsigned int i : fe_values.dof_indices())
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }


  bound gamaF;
  gamaF.set_time(0);
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           gamaF,
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}



void Step3::solve()
{
  SolverControl solver_control(2000, 1e-6);
  SolverCG<Vector<double>> solver(solver_control);

  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}



void Step3::output_results() const
{
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();

  std::ofstream output("solutionLap.vtu");
  data_out.write_vtu(output);
  std::ofstream solDat("sol.dat");
  for (int i=0; i<solution.size() ; ++i)
  solDat << solution(i) << " ";
  std::cout << "size of solution is " << solution.size() <<std::endl;
}



void Step3::run()
{
  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}



int main()
{
  deallog.depth_console(2);

  Step3 laplace_problem;
  laplace_problem.run();

  return 0;
}
