#include <cstdio>
#include <cmath>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_matrix.h>

// Example slow–fast system (van der Pol type), with parameter epsilon
int func(double t, const double y[], double f[], void *params) {
    double epsilon = *(double*)params;
    // y[0] = x, y[1] = y
    f[0] = y[1];
    f[1] = (1.0 / epsilon) * ( (1 - y[0]*y[0]) * y[1] - y[0] );
    return GSL_SUCCESS;
}

int jac(double t, const double y[], double *dfdy, double dfdt[], void *params) {
    double epsilon = *(double*)params;
    gsl_matrix_view m = gsl_matrix_view_array(dfdy, 2, 2);
    gsl_matrix *J = &m.matrix;

    gsl_matrix_set(J, 0, 0, 0.0);
    gsl_matrix_set(J, 0, 1, 1.0);

    double a = (1.0/epsilon) * (-2.0 * y[0] * y[1] - 1.0);
    double b = (1.0/epsilon) * (1.0 - y[0]*y[0]);
    gsl_matrix_set(J, 1, 0, a);
    gsl_matrix_set(J, 1, 1, b);

    dfdt[0] = 0.0;
    dfdt[1] = 0.0;
    return GSL_SUCCESS;
}

int main() {
    double epsilon = 0.01;            // small parameter (adjust to examine canards)
    gsl_odeiv2_system sys = {func, jac, 2, &epsilon};

    gsl_odeiv2_driver *d =
        gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45,
                                      1e-6, 1e-6, 0.0);

    double t = 0.0;
    double t1 = 100.0;               // integrate until t1
    double y[2] = { 1.0, 0.0 };      // initial conditions (x0, y0)

    for (int i = 1; i <= 1000; i++) {
        double ti = i * t1 / 1000.0;
        int status = gsl_odeiv2_driver_apply(d, &t, ti, y);

        if (status != GSL_SUCCESS) {
            printf("Error, return value=%d\n", status);
            break;
        }
        printf("%.5e %.5e %.5e\n", t, y[0], y[1]);
    }

    gsl_odeiv2_driver_free(d);
    return 0;
}

