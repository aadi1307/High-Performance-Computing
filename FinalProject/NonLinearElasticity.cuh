#ifndef NLELASTICITY_CUH
#define NLELASTICITY_CUH
#include <string>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/IterativeLinearSolvers"
 
//
// Created by egi on 9/15/19.
//

#ifndef MATRIX_FORMAT_PERFORMANCE_RESIZABLE_GPU_MEMORY_H
#define MATRIX_FORMAT_PERFORMANCE_RESIZABLE_GPU_MEMORY_H

#include <memory>

void internal_resize (char *& data, size_t new_size);
void internal_free (char *& data);

template <typename T>
class resizable_gpu_memory
{
public:
  resizable_gpu_memory () = default;
  ~resizable_gpu_memory () { internal_free (data); }

  resizable_gpu_memory (const resizable_gpu_memory &) = delete;
  resizable_gpu_memory &operator= (const resizable_gpu_memory &) = delete;

  void clear ()
  {
    internal_free (data);
    size = 0;
  }

  void resize (size_t new_size)
  {
    if (new_size > size)
    {
      size = new_size;
      internal_resize (data, size * sizeof (T));
    }
  }

  T *get () { return reinterpret_cast<T*> (data); }
  const T *get () const { return reinterpret_cast<const T*> (data); }

private:
  size_t size = 0;
  char *data = nullptr;
};

#endif // MATRIX_FORMAT_PERFORMANCE_RESIZABLE_GPU_MEMORY_H






//
// Created by egi on 10/31/19.
//

#ifndef MATRIX_FORMAT_PERFORMANCE_MEASUREMENT_CLASS_H
#define MATRIX_FORMAT_PERFORMANCE_MEASUREMENT_CLASS_H

#include <string>
#include <iostream>

class measurement_class
{
  const size_t giga = 1E+9;

public:
  measurement_class () = default;
  measurement_class (
      std::string format,
      double elapsed,
      double load_store_bytes,
      double operations_count);

  double get_elapsed () const { return elapsed; }
  double get_effective_bandwidth () const { return effective_bandwidth; }
  double get_computational_throughput () const { return computational_throughput; }

  const std::string &get_format () const { return matrix_format; }

  measurement_class & operator+=(const measurement_class &rhs)
  {
    elapsed += rhs.get_elapsed ();
    effective_bandwidth += rhs.get_effective_bandwidth ();
    computational_throughput += rhs.get_computational_throughput ();

    matrix_format = rhs.get_format ();
    measurements_count++;

    return *this;
  }

  void finalize ()
  {
    if (measurements_count)
    {
      elapsed /= measurements_count;
      effective_bandwidth /= measurements_count;
      computational_throughput /= measurements_count;
    }
  }

private:
  double elapsed {};
  double effective_bandwidth {};
  double computational_throughput {};
  std::string matrix_format;

  unsigned int measurements_count {};
};

template <typename data_type>
void compare_results (unsigned int y_size, const data_type *a, const data_type *b)
{
  data_type numerator = 0.0;
  data_type denumerator = 0.0;

  for (unsigned int i = 0; i < y_size; i++)
  {
    numerator += (a[i] - b[i]) * (a[i] - b[i]);
    denumerator += b[i] * b[i];
  }

  const data_type error = numerator / denumerator;

  if (error > 1e-9)
  {
    std::cerr << "ERROR: " << error << std::endl;

    for (unsigned int i = 0; i < y_size; i++)
    {
      if (std::abs (a[i] - b[i]) > 1e-8)
      {
        std::cerr << "a[" << i << "] = " << a[i] << "; b[" << i << "] = " << b[i] << std::endl;
        break;
      }
    }
  }

  std::cerr.flush ();
}


#endif // MATRIX_FORMAT_PERFORMANCE_MEASUREMENT_CLASS_H






//
// Created by egi on 9/14/19.
//

#ifndef MATRIX_MARKET_READER_H
#define MATRIX_MARKET_READER_H

#include <istream>
#include <memory>

namespace matrix_market
{

class matrix_class
{
public:

  enum class format : int
  {
    coordinate,  ///< Sparse matrices
    array        ///< Dense matrices
  };

  enum class data_type : int
  {
    integer, real, complex, pattern
  };

  enum class storage_scheme : int
  {
    general, symmetric, hermitian, skew_symmetric
  };

  class matrix_meta
  {
  public:
    const unsigned int rows_count = 0;
    const unsigned int cols_count = 0;
    const unsigned int non_zero_count = 0;

    const format matrix_format;
    const data_type matrix_data_type;
    const storage_scheme  matrix_storage_scheme;

  public:
    matrix_meta (
        unsigned int rows_count_arg,
        unsigned int cols_count_arg,
        unsigned int non_zero_count_arg,

        format matrix_format_arg,
        data_type matrix_data_type_arg,
        storage_scheme  matrix_storage_scheme_arg)
      : rows_count (rows_count_arg)
      , cols_count (cols_count_arg)
      , non_zero_count (non_zero_count_arg)
      , matrix_format (matrix_format_arg)
      , matrix_data_type (matrix_data_type_arg)
      , matrix_storage_scheme (matrix_storage_scheme_arg)
    { }

    bool is_sparse () const { return matrix_format == format::coordinate; }
    bool is_dense () const { return !is_sparse (); }
  };

  matrix_class () = delete;
  explicit matrix_class (matrix_meta meta_arg) : meta (meta_arg) { }

  size_t get_rows_count () const { return meta.rows_count; }
  size_t get_cols_count () const { return meta.cols_count; }

  virtual const unsigned int *get_row_ids () const = 0; /// Return nullptr for dense matrices
  virtual const unsigned int *get_col_ids () const = 0; /// Return nullptr for dense matrices
  virtual const void *get_data () const = 0;

  const double *get_dbl_data () const
  {
    if (meta.matrix_data_type == data_type::real)
      return reinterpret_cast<const double *> (get_data ());
    throw std::runtime_error ("Accessing non dbg matrix as dbl");
  }

  const int *get_int_data () const
  {
    if (meta.matrix_data_type == data_type::integer)
      return reinterpret_cast<const int*> (get_data ());
    throw std::runtime_error ("Accessing non int matrix as int");
  }

public:
  const matrix_meta meta;
};

class reader
{
public:
  reader () = delete;
  explicit reader (std::istream &is, bool throw_exceptions=true);

  operator bool () const;

  matrix_class &matrix ();
  const matrix_class &matrix () const;

private:
  bool is_correct = false;
  std::unique_ptr<matrix_class> matrix_data;
};

}

#endif // MATRIXMARKETREADER_MATRIX_MARKET_READER_H









//
// Created by egi on 9/15/19.
//

#ifndef MATRIX_FORMAT_PERFORMANCE_MATRIX_CONVERTER_H
#define MATRIX_FORMAT_PERFORMANCE_MATRIX_CONVERTER_H

#include <memory>

struct matrix_rows_statistic
{
  unsigned int min_elements_in_rows {};
  unsigned int max_elements_in_rows {};
  unsigned int avg_elements_in_rows {};
  double elements_in_rows_std_deviation {};
};

matrix_rows_statistic get_rows_statistics (
    const matrix_market::matrix_class::matrix_meta &meta,
    const unsigned int *row_ptr);

template <typename data_type>
class csr_matrix_class
{
public:
  csr_matrix_class () = delete;
  explicit csr_matrix_class (const matrix_market::matrix_class &matrix, bool row_ptr_only=false);

  const matrix_market::matrix_class::matrix_meta meta;

  size_t get_matrix_size () const;

public:
  std::unique_ptr<data_type[]> data;
  std::unique_ptr<unsigned int[]> columns;
  std::unique_ptr<unsigned int[]> row_ptr;
};

template <typename data_type>
class ell_matrix_class
{
public:
  ell_matrix_class () = delete;
  explicit ell_matrix_class (csr_matrix_class<data_type> &matrix);
  ell_matrix_class (csr_matrix_class<data_type> &matrix, unsigned int elements_in_row_arg);

  static size_t estimate_size (csr_matrix_class<data_type> &matrix);

  const matrix_market::matrix_class::matrix_meta meta;

  size_t get_matrix_size () const;

public:
  std::unique_ptr<data_type[]> data;
  std::unique_ptr<unsigned int[]> columns;

  unsigned int elements_in_rows = 0;
};

template <typename data_type>
class coo_matrix_class
{
public:
  coo_matrix_class () = delete;
  explicit coo_matrix_class (csr_matrix_class<data_type> &matrix);
  coo_matrix_class (csr_matrix_class<data_type> &matrix, unsigned int element_start);

  const matrix_market::matrix_class::matrix_meta meta;

  size_t get_matrix_size () const;

public:
  std::unique_ptr<data_type[]> data;
  std::unique_ptr<unsigned int[]> cols;
  std::unique_ptr<unsigned int[]> rows;

private:
  size_t elements_count {};
};

template <typename data_type>
class hybrid_matrix_class
{
public:
  hybrid_matrix_class () = delete;
  explicit hybrid_matrix_class (csr_matrix_class<data_type> &matrix);

  void allocate (csr_matrix_class<data_type> &matrix, double percent);

  const matrix_market::matrix_class::matrix_meta meta;

  std::unique_ptr<ell_matrix_class<data_type>> ell_matrix;
  std::unique_ptr<coo_matrix_class<data_type>> coo_matrix;
};

#endif // MATRIX_FORMAT_PERFORMANCE_MATRIX_CONVERTER_H









__global__ void print_kernel();
template <typename data_type>
__global__ void csr_adaptive_spmv_kernel (
    const unsigned int n_rows,
    const int *col_ids,
    const int *row_ptr,
    const unsigned int *row_blocks,
    const data_type *data,
    const data_type *x,
    data_type *y);

template <typename data_type>
__global__ void csr_spmv_kernel(
    unsigned int n_rows, 
    const int *col_ids, 
    const int *row_ptr, 
    const data_type *data, 
    const data_type *x, 
    data_type *y);

#define FULL_WARP_MASK 0xFFFFFFFF
template <class T>
__device__ T warp_reduce (T val);


template <typename data_type>
__global__ void csr_spmv_vector_kernel (
unsigned int n_rows, 
const int *col_ids, 
const int *row_ptr, 
const data_type *data, 
const data_type *x, 
data_type *y);


class NLElasticity {
    
     public:
      
 __host__ void print();
       
        
        NLElasticity();  // declare default constructor
        NLElasticity(const std::string& parameter1, int parameter2, int parameter3, int parameter4, 
                           double parameter5, double parameter6, double parameter7);
        std::string meshType;
        std::string materialModel;
        int nElemX ;
        int nElemY ;
        int nElemZ ;
        double sizeX;
        double sizeY;
        double sizeZ;
        void createHexGridMeshGeneral(int nElemX, int nElemY, int nElemZ, double sizeX, double sizeY, double sizeZ);

        std::string myShape;          // Linear or quadratic shape functions
        int myNumNodes;               // Total number of nodes
        int myNumElems;               // Total number of elements
        int myNodesPerSurface;
        int myNCoord;               //cooordinates = 3 for 3D..
        double myElemSize;           // Common element size determined from total length and #elems
        double myElementLength;      // Length of each element
        int myNodesPerElement;       // 2 for linear, 3 for quadratic element
        std::string myElemShape;     // Element shape (hex, etc.)
        bool myApplyFiniteStrain;
        bool myVerbose;
        int myNonlinearSteps;
        double myE; // Young's modulus
        double myNu; // Poisson's ratio
        std::string myMaterialModel; // Material model
        double myBulkModulus; // Bulk modulus
        double myMu; // Shear modulus
        std::vector<std::vector<int>> myBCtype;
        std::vector<std::vector<double>> myBCvalue;

        std::vector<double> myBodyForce; // Vector for body force
        Eigen::VectorXd mySol;
        Eigen::VectorXd myDeltaSol;
       
        int myDOFPerElem;                 // Degrees of freedom per element
        int myDOFPerNode;                 // Degrees of freedom per element
        //Eigen::MatrixXd myBCvalue;       // Boundary condition values
        std::vector<std::vector<double> > myN; // shape functions
        
        // Define myGradN as a vector of matrices
        std::vector<Eigen::MatrixXd> myGradN;


        std::vector<std::vector<double> > myXi; // local coordinate at quadrature point
        std::vector<double> myWt; // weight
        // Define the struct for myMesh
        struct myMesh {
            std::vector<std::vector<double> > p; // Nodal coordinate matrix
            std::vector<std::vector<int> > q;    // Element connectivity matrix
            std::vector<std::vector<int> > e;    // Boundary matrix (added from previous code)
        };

        myMesh mesh;

        //Eigen::VectorXd mySol;
        Eigen::VectorXd myU;
        Eigen::VectorXd myV;
        Eigen::VectorXd myW;
        Eigen::VectorXd myDeformation;
        double myMaxDelta;
        Eigen::MatrixXd myK;
        Eigen::MatrixXd myF;
        void ShapeFunction2D(const std::vector<double>& xi,
                       std::vector<double>& N,
                       std::vector<std::vector<double>>& gradN);
        std::pair<std::vector<std::vector<double> >, std::vector<std::vector<double> > > ShapeFunction(const std::vector<double>& xi);
        std::vector<double> myNodalDirichlet; // Vector for Dirichlet boundary conditions
        std::vector<int> myFixedDOF; // List of fixed degrees of freedom
        std::vector<int> myFixedNodes; // List of fixed nodes
        std::vector<int> myFreeDOF;

       void applyDirichletOnSurfaceFunction(const std::string& uString,
                                        const std::string& vString,
                                        const std::string& wString,
                                        int EdgeNum);
 
        // Function to compute the Jacobian for an element
        std::vector<std::vector<double> > Jacobian(int elem, const std::vector<double> xi);
            // Function to compute the Jacobian with provided nodes
        std::vector<std::vector<double> > JacobianWithNodes(
            const std::vector<std::vector<double> >& positionNodes, const std::vector<double> xi);
        // Function to get the position of nodes
        std::vector<std::vector<double> > GetNodesPosition(const std::vector<int>& nodes);

        //function to compute the determinant of the Jacobian
        std::vector<std::vector<double>> myD; // Material stiffness matrix

        //function to get the material parameter
        void setMaterialParameter(double E);

        //function to get the youngs modulus
        void setYoungsModulus(double E);

        //function to get the poisson ratio
        void setPoissonsRatio(double nu);

        //setting the material model
        void setMaterialModel(const std::string& materialModel);

        //function to get bulk modulus
        void setBulkModulus(double K1);

        //function to get shear modulus
        void setShearModulus(double mu1);

        //function to compute the D matrix
        void computeDMatrix();

        //function for Dirichlet boundary conditions on DOF
        void applyDirichletOnDOF(double value, int dof);

        //function for Dirichlet boundary conditions on edge
        void applyDirichletOnSurface(double value, int SurfaceNum);
        //function for Neumann boundary condition on a given edge
        void applyNeumanOnSurface(double forceU, double forceV, double forceW, size_t SurfaceNum);
        static const int MAX_SURFACE_NUM = 100;  // Adjust the size accordingly
        static const int NUM_BC_TYPES = 3;  // Assuming there are three types (0, 1, 2), adjust as needed

        void printEigenMatrix(const Eigen::MatrixXd& matrix, const std::string& matrixName);
        void GaussQuad2dQuad(int numGQ, std::vector<double>& xi_GQ, std::vector<double>& wt_GQ);
        void GaussQuad3dHex(int numGQ, std::vector<std::vector<double>>& xi_GQ, std::vector<double>& wt_GQ);
        void printVectorOfVectors(const std::vector<std::vector<double>>& vec);

        void lgwt(int N, double a, double b, std::vector<double>& x, std::vector<double>& w);
        void assembleBC();

        std::vector<double> myfC;
        int myNumDOF;
        std::vector<double> myForcedNodes;
        Eigen::VectorXd myNeumanForce;
         // Declaration for myNumBoundaries
        int myNumBoundaries;
        double energyErrNorm;
        double L2ErrNorm;
        double RelativeL2ErrNorm;
        double RelativeEnergyErrNorm;
        int myProblem;
        int myDomainLength;
        Eigen::VectorXd myElemVol;

       std::vector<double> integrateOverBoundary(int geomEdge, int seg, const std::vector<double>& wt_GQ,
                                 const std::vector<Eigen::MatrixXd>& N2D,
                                 const std::vector<Eigen::MatrixXd>& gradN2D, int dof);
        std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix);
        std::vector<std::vector<double>> multiplyMatrix(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2);
        double determinant(const std::vector<std::vector<double>>& matrix);
      
    std::vector<std::vector<double>> eigenMatrixToStdVector(const Eigen::MatrixXd& matrix);
    double trace(const std::vector<std::vector<double>>& matrix);

    std::vector<std::vector<std::vector<std::vector<double>>>> computeElasticityTensorGeneralizedNeoHookean(
        const std::string& materialModel, int NCoords, double K1, double mu1, const std::vector<std::vector<double>>& B, double J);
    void computeElementStiffnessFiniteStrainSpatialConf(const int elem, Eigen::MatrixXd& KElem, Eigen::VectorXd& fElem);
       Eigen::MatrixXd KirchhoffStress(const std::string& materialModel, int NCoords, double K1, double mu1, const Eigen::MatrixXd& B, double J);
       void assembleK();
    
    Eigen::VectorXd conjugateGradientSolver(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &B);
    void solveNonlinearFEM();
    double getConditionNumber(const Eigen::MatrixXd& matrix);

    void solveFEMProblem();


};

#endif
