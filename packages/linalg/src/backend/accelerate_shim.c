#define ACCELERATE_NEW_LAPACK 1
#include <vecLib/lapack.h>

void linalg_sgesdd(char jobz, int m, int n, float *a, int lda, float *s,
                   float *u, int ldu, float *vt, int ldvt, float *work,
                   int lwork, int *iwork, int *info) {
    sgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
            iwork, info);
}

void linalg_dgesdd(char jobz, int m, int n, double *a, int lda, double *s,
                   double *u, int ldu, double *vt, int ldvt, double *work,
                   int lwork, int *iwork, int *info) {
    dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
            iwork, info);
}

void linalg_cgesdd(char jobz, int m, int n, void *a, int lda, float *s,
                   void *u, int ldu, void *vt, int ldvt, void *work,
                   int lwork, float *rwork, int *iwork, int *info) {
    cgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
            rwork, iwork, info);
}

void linalg_zgesdd(char jobz, int m, int n, void *a, int lda, double *s,
                   void *u, int ldu, void *vt, int ldvt, void *work,
                   int lwork, double *rwork, int *iwork, int *info) {
    zgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
            rwork, iwork, info);
}
