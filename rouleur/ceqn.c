#include <Python.h>
#include <math.h>

// https://docs.python.org/3.5/extending/extending.html

static PyObject *
martin_eqn_compiled(PyObject *self, PyObject *args, PyObject *keywds)
{
	// A more efficient, though visually disgusting, implementation of the
	// `martin_eqn` function provided in the `internals` module.

	// NOTE: Will only handle scalar arguments!

	// Need to handle a lot of arguments initially, which is inevitably
	// a copy+paste job. After which, code can more or less be transplanted
	// from the python function.

	// INPUTS:
	double power_output,
         rider_velocity,
         air_velocity,
         air_density,
         CdA,
         Fw,
         Crr,
         mass_total,
         road_gradient,
         chain_efficiency_factor,
         g,
         wind_velocity,
         wind_direction,
         travel_direction,
         initial_rider_velocity,
         final_rider_velocity,
         delta_time,
         wheel_inertia,
         tire_radius;

  // COMPONENTS:
	double air_resistance,
				 rolling_resistance,
				 wheel_friction,
			   potential_energy_changes,
			   kinetic_energy_changes;

	// RETURNED VALUE:
	double out;

	static char *kwlist[] = {"power_output",
									         "rider_velocity",
									         "air_velocity",
									         "air_density",
									         "CdA",
									         "Fw",
									         "Crr",
									         "mass_total",
									         "road_gradient",
									         "chain_efficiency_factor",
									         "g",
									         "wind_velocity",
									         "wind_direction",
									         "travel_direction",
									         "initial_rider_velocity",
									         "final_rider_velocity",
									         "delta_time",
									         "wheel_inertia",
									         "tire_radius",
									         NULL};

	if (!PyArg_ParseTupleAndKeywords(
				args, keywds, "ddddddddddddddddddd", kwlist,

				&power_output,
				&rider_velocity,
				&air_velocity,
				&air_density,
				&CdA,
				&Fw,
				&Crr,
				&mass_total,
				&road_gradient,
				&chain_efficiency_factor,
				&g,
				&wind_velocity,
				&wind_direction,
				&travel_direction,
				&initial_rider_velocity,
				&final_rider_velocity,
				&delta_time,
				&wheel_inertia,
				&tire_radius))
		return NULL;

  air_resistance = (
  	pow(air_velocity, 2)
	  * rider_velocity
	  * 0.5
	  * air_density
	  * (CdA + Fw)
  );

  rolling_resistance = (
  	rider_velocity
		* Crr
		* mass_total
		* g
		* cos(atan(road_gradient))
  );

  wheel_friction = (
  	rider_velocity
    * (91 + 8.7 * rider_velocity)
    * 0.001
  );

  potential_energy_changes = (
  	rider_velocity
		* mass_total
		* g
		* sin(atan(road_gradient))
  );

  kinetic_energy_changes = (
  	0.5
	  * (mass_total + wheel_inertia / pow(tire_radius, 2))
	  * (pow(final_rider_velocity, 2) - pow(initial_rider_velocity, 2))
	  / delta_time
  );

  out = (
  	air_resistance
  	+ rolling_resistance
  	+ wheel_friction
  	+ potential_energy_changes
  	+ kinetic_energy_changes
  ) / chain_efficiency_factor;

  return Py_BuildValue("f", out);
}

// Module Method Table
// -------------------
static PyMethodDef RouleurMethods[] = {
    {"martin_eqn_compiled",
     (PyCFunction)martin_eqn_compiled, METH_VARARGS | METH_KEYWORDS,
     "A compiled version of rouleur.pyeqn.martin_eqn; takes only scalar arguments."},

    {NULL, NULL, 0, NULL}   // sentinel
};

// Module Definition Structure
// ---------------------------
static struct PyModuleDef rouleurmodule = {
   PyModuleDef_HEAD_INIT,
   "ceqn",          // name of the module (used in setup.py)
   NULL,            // doc string, may be NULL
   -1,              // -1 if the module keeps state in global variables
   RouleurMethods
};

// Module Initialization Function
// ------------------------------
PyMODINIT_FUNC
PyInit_ceqn(void)    // name is important!
{
  return PyModule_Create(&rouleurmodule);
};
