#include <vecLib/cblas.h>

void linalg_sgesdd(char jobz, int m, int n, float *a, int lda, float *s,
                   float *u, int ldu, float *vt, int ldvt, float *work,
                   int lwork, int *iwork, int *info);
void linalg_dgesdd(char jobz, int m, int n, double *a, int lda, double *s,
                   double *u, int ldu, double *vt, int ldvt, double *work,
                   int lwork, int *iwork, int *info);
void linalg_cgesdd(char jobz, int m, int n, void *a, int lda, float *s,
                   void *u, int ldu, void *vt, int ldvt, void *work,
                   int lwork, float *rwork, int *iwork, int *info);
void linalg_zgesdd(char jobz, int m, int n, void *a, int lda, double *s,
                   void *u, int ldu, void *vt, int ldvt, void *work,
                   int lwork, double *rwork, int *iwork, int *info);

void linalg_sgelss(int m, int n, int nrhs, float *a, int lda, float *b,
                   int ldb, float *s, float rcond, int *rank, float *work,
                   int lwork, int *info);
void linalg_dgelss(int m, int n, int nrhs, double *a, int lda, double *b,
                   int ldb, double *s, double rcond, int *rank,
                   double *work, int lwork, int *info);
void linalg_cgelss(int m, int n, int nrhs, void *a, int lda, void *b,
                   int ldb, float *s, float rcond, int *rank, void *work,
                   int lwork, float *rwork, int *info);
void linalg_zgelss(int m, int n, int nrhs, void *a, int lda, void *b,
                   int ldb, double *s, double rcond, int *rank, void *work,
                   int lwork, double *rwork, int *info);
