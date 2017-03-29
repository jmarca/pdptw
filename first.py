from __future__ import print_function
from ortools.linear_solver import pywraplp

def main():
  # Instantiate a mixed-integer solver, naming it SolveIntegerProblem.
  # solver = pywraplp.Solver('SolveIntegerProblem',
  #                          pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

  solver = pywraplp.Solver('SolveSimpleSystem',
                           pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
  # x and y are integer non-negative variables.
  # x = solver.IntVar(0.0, solver.infinity(), 'x')
  # y = solver.IntVar(0.0, solver.infinity(), 'y')

  x = solver.NumVar(-solver.infinity(), solver.infinity(), 'x')
  y = solver.NumVar(-solver.infinity(), solver.infinity(), 'y')


  # x + 7 * y <= 17.5
  constraint1 = solver.Constraint(-solver.infinity(), 17.5)
  constraint1.SetCoefficient(x, 1)
  constraint1.SetCoefficient(y, 7)

  # x <= 3.5
  constraint2 = solver.Constraint(0.0, 3.5)
  constraint2.SetCoefficient(x, 1)
  constraint2.SetCoefficient(y, 0)

  # Maximize x + 10 * y.
  objective = solver.Objective()
  objective.SetCoefficient(x, 1)
  objective.SetCoefficient(y, 10)
  objective.SetMaximization()

  """Solve the problem and print the solution."""
  result_status = solver.Solve()
  # The problem has an optimal solution.
  assert result_status == pywraplp.Solver.OPTIMAL

  # The solution looks legit (when using solvers other than
  # GLOP_LINEAR_PROGRAMMING, verifying the solution is highly recommended!).
  assert solver.VerifySolution(1e-7, True)

  print('Number of variables =', solver.NumVariables())
  print('Number of constraints =', solver.NumConstraints())

  # The objective value of the solution.
  print('Optimal objective value = %.2f' % solver.Objective().Value())
  print()
  # The value of each variable in the solution.
  variable_list = [x, y]

  for variable in variable_list:
      print('%s = %.2f' % (variable.name(), variable.solution_value()))

if __name__ == '__main__':
  main()
