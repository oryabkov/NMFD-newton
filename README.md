
# Basic interfaces

The following is the list of Named Conventions (like Iterator in STL) often used in NMFD. Perhaps sometime in the future we could add Concepts for these (however c++20 is needed which is not ok for now). Note that not every single aspect is specified thoroughly so futher specification of some details is needed.

## VectorOperations + VectorSpace

TODO

VectorOperations is basically BLAS1 interface.

```
/// Nested types
using scalar_type = ...;
using vector_type = ...;

/// Methods

/// y<-x
void assign(const vector_type& x, vector_type& y) const

void assign_scalar(const scalar_type val, vector_type& x)const 
/// x<-mul_x*x+scalar
void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x)

void assign_random(vector_type& x)const 
/// x<-x*scale
void scale(const scalar_type scale, vector_type& x)
/// returns sum of all elements
[[nodiscard]] scalar_type sum(const vector_type& x)const 
/// returns sum of all elements absolute values
[[nodiscard]] scalar_type asum(const vector_type &x)const
/// returns either all elements are valid numbers (not nans or infs)
[[nodiscard]] bool is_valid_number(const vector_type &x)const
/// returns inner product of vectors (sum of elements products)
[[nodiscard]] scalar_type scalar_prod(const vector_type &x, const vector_type &y)const
/// returns some weighted inner product of vectors (problem dependent, for example L2 integral norm)
[[nodiscard]] scalar_type scalar_prod_l2(const vector_type &x, const vector_type &y)const
/// Returns ANY norm that fits problem convergence
[[nodiscard]] scalar_type norm(const vector_type &x) const
/// norm(x)^2
[[nodiscard]] scalar_type norm_sq(const vector_type &x) const
/// scalar_prod(x,x)
[[nodiscard]] scalar_type norm2_sq(const vector_type &x) const
/// sqrt(scalar_prod(x,x))
[[nodiscard]] scalar_type norm2(const vector_type &x) const
/// scalar_prod_l2(x,x)
[[nodiscard]] scalar_type norm_l2_sq(const vector_type &x) const
/// sqrt(scalar_prod_l2(x,x))
[[nodiscard]] scalar_type norm_l2(const vector_type &x) const
/// asum(x)
[[nodiscard]] scalar_type norm1(const vector_type &x) const
/// returns some weighted L1/l1 norm (problem dependent)
[[nodiscard]] scalar_type norm_l1(const vector_type &x) const
/// returns maximum of all elements absolute values
[[nodiscard]] scalar_type norm_inf(const vector_type &x) const
/// returns maximum of all elements absolute weighted values (problem dependent)
[[nodiscard]] scalar_type norm_l_inf(const vector_type &x) const
/// May be ommited for now
[[nodiscard]] vector_type at(multivector_type& x, ordinal_type m, ordinal_type k_)
/// y<-x*mul_x+y*mul_y
void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y) const
/// z<-x*mul_x+y*mul_y+z*mul_z
void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, const scalar_type mul_z, vector_type& z) const
/// y<-x*mul_x
void assign_lin_comb(const scalar_type mul_x, const vector_type& x, vector_type& y) const
/// z<-x*mul_x+y*mul_y
void assign_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, vector_type& z) const
```

vector_type supposed to be DefaultConstructable.

VectorSpace has additional methods that allocate/deallocate vectors meaning that VectorSpace 'knows' 'size' of the vector space:

```
void init_vector(vector_type &v);
void start_use_vector(vector_type &v);
void stop_use_vector(vector_type &v);
void free_vector(vector_type &v);
```

NOTE: VectorSpace supposed to have all the operations VectorOperations have but it is not streaktly speaking restriction of VectorOperations (i.e. not every VectorSpace HAS to be VectorOperations). The difference is VectorSpace MUST to operate only on the vectors that were merged by its instance (i.e. vectors of particlar size). Whilest VectorOperations supposed to work with ANY correctly initialized vector of vector_type. If however VectorSpace implements such universality it doesnot contradict its convention. Good example of VectorSpace that is not VectorOperations is space that uses plain pointer as its vector_type. VectorSpace may use its limited usability to optimize memory consumption or efficiency (for example preallocating some internal buffers with already known size).

Rationale: why do we need two pairs of basically alloc/dealloc methods (init/free,start_use/stop_use)?
init/free pair is supposed to be used in constructor/destructor of algorithms to signal that we possibly need to use such a additional vectors.
start_use/stop_use pair is supposed to be used around actual vector usage code segment.
start_use/stop_use pair garantees that between these calls vector values won't be lost and we can actually use them. Simplest way is to call this pair in the begining and end of method that implements an algorithm (like solve method). But may be used in more delicate way inside algorithm - to mark regions where temporal buffer values are actually needed.

This conventions allow user to implement different strategies of memory managment. If, for example, memory capacity is not a problem but we want to accelerate our calculation by reducing the number of allocation calls (which can be rather expensive, for example, for CUDA) - we may implement init/free strategy for allocation/deallocation. This also gives us benefit: if there is not enough memory in the system for algorithm to run we will know it right after calc starts - instead of getting not enough memory after hours of calculations.
On the other hand if memory storage is our priority we may use start_use/stop_use pair to allocate deallocate memory - to allow reusage of the memory between different parts in different algorithms - or ever reusage of memory inside single algorithm if start_use/stop_use pair is well embedded into logic. This strategy may be combined with well written memory manager that utulizes the fact that all vectors in space have the same lenght - so we may implement some sort of pool of vectors or 'equal elements' memory manager to accelerate memory allocation. 
In another scenario we may allocate memory once in start_use_vector if vectors was not allocated yet and do nothing otherwise. In this case we may use free_vector to free memomry if it was allocated earlier - or use RAII objects as vectors. In this case all memory will be allocated once when algorithm starts without futher reallocations - but if some algorithms for some reason are not invoked in current calculation memory wont be allocated. Another important case - adaptive mesh algorithms. In such a case we may proceed some time with vectors of one size and then suddenly - when mesh adapts - need to reallocate buffers. This may be performed inside start_use_vector by comparing the size of the current buffer already allocated with the space vector size - and if they differ - reallocate buffer. 
Any way i beleive there are different situation and there is no one 'absolutly correct' strategy, while this overwhelming interface allows user to decide which is correct in current case.

## MatrixVectorOperations + MatrixMatrixOperations + LinalOperations

TODO 

MatrixVectorOperations is basically BLAS2 interface.
MatrixMatrixOperations is basically BLAS3 interface.

LinalOperations (or simply Operations when there is no confusion with something else) is 
simply combination of VectorOperations, MatrixVectorOperations, MatrixMatrixOperations. In other words is BLAS synonim.

## Operator 

```
class OperatorName
{
public:
    using scalar_type = ...;
    using vector_type = ...;
public:
    void apply(const vector_type &in, vector_type &out);
};
```

In Operator there is a garantee that initial value of out doesnot affects result.

## InplaceOperator 

```
class InplaceOperatorName
{
public:
    using scalar_type = ...;
    using vector_type = ...;
public:
    void apply(vector_type &in_out);
};
```

in_out is both input and output.

## OperatorWithSpaces

```
class OperatorWithSpacesName
{
public:
    using scalar_type = ...;
    using vector_type = ...;
    using vector_space_type = ...; /// VectorSpace
public:
    void apply(const vector_type &in, vector_type &out);
    std::shared_ptr<vector_space_type> get_im_space() const;
    std::shared_ptr<vector_space_type> get_dom_space() const;
};
```

In Operator there is a garantee that initial value of out doesnot affects result.

## InplaceOperatorWithSpaces 

```
class InplaceOperatorWithSpacesName
{
public:
    using scalar_type = ...;
    using vector_type = ...;
    using vector_space_type = ...; /// VectorSpace
public:
    void apply(vector_type &in_out);
    std::shared_ptr<vector_space_type> get_im_space() const;
    std::shared_ptr<vector_space_type> get_dom_space() const;
};
```

in_out is both input and output.

## OperatorWithDerivative

```
class OperatorName
{
public:
    using scalar_type = ...;
    using vector_type = ...;
    using jacobi_operator_type = ...;
public:
    void apply(const vector_type &in, vector_type &out);

    void set_linearization_point(const vector_type &p);
    const std::shared_ptr<const jacobi_operator_type> &get_jacobi_operator();
};
```

## Preconditioner

Preconditioner is restriction of Operator and InplaceOperator. Plus it has set_operator method for inicialization/reinitialization.

```
class PreconditionerName
{
public:
    using scalar_type = ...;
    using vector_type = ...;
    using operator_type = ...;
public:
    void set_operator(std::shared_ptr<const operator_type> operator);
    void apply(const vector_type &in, vector_type &out);
    void apply(vector_type &in_out);
};
```

## Solver 

```
class SolverName
{
public:
    using vector_type = ...;
    using operator_type = ...;
public:
    void set_operator(std::shared_ptr<const operator_type> operator);
    bool solve(const vector_type &rhs, vector_type &res);
};
```

In Solver there is no strict garantee that initial value of res doesnot affects result. For example Solver can use it initial guess.

## PreconditionerWithSpaces, SolverWithSpaces

TODO seems not be often used though

```
class PreconditionerName
{
public:
    using scalar_type = ...;
    using vector_type = ...;
    using operator_type = ...;
    using vector_space_type = ...; /// VectorSpace
public:
    void set_operator(std::shared_ptr<const operator_type> operator);
    void apply(const vector_type &in, vector_type &out);
    void apply(vector_type &in_out);
    std::shared_ptr<vector_space_type> get_im_space() const;
    std::shared_ptr<vector_space_type> get_dom_space() const;
};
```

## Coarsening

Creates Restrictor,Prolongator and SystemOperator for next level based on previous SystemOperator.

```
class CoarseningName
{
public:
    using operator_type = ...;
    using restrictor_type = ...;
    using prolongator_type = ...;
public:
    std::tuple<std::shared_ptr<restrictor_type>,std::shared_ptr<prolongator_type>> next_level(const operator_type &op);
    std::shared_ptr<operator_type> coarse_operator(const operator_type &op, const restrictor_type &restrictor, const prolongator_type &prolongator);
    bool coarse_enough(const operator_type &op)const;
};
```

This structure is almost like in amgcl.
Rationale: why do we need two methods for level creation? because of possible rebuild procedure where only operators are updated.

TODO

## HierarchicAlgorithm

Used for automized nested algorithms construction.

```
class HierarchicAlgorithmName
{
public:
    using params_hierarchy = ...;
    using utils_hierarchy = ...;

    HierarchicAlgorithmName(const utils_hierarchy &u, const params_hierarchy &p)
};
```

Also params_hierarchy and utils_hierarchy supposed to be DefaultConstructable 
Plus for now params_hierarchy must have that following constructors:
```
params_hierarchy(const std::string &log_prefix);
params_hierarchy(const std::string &log_prefix, const std::string &log_name);
```
Meaning simpliest way to write constructor is:
```
params_hierarchy(const std::string &log_prefix = "", const std::string &log_name = "");
```
This additional convention helps to construct nested prefixes for algorithms in logs and exceptions (like newton::linsolver(gmres)::prec(mg)::...) but seems too much as mandatory so it is debatable.

HierarchicAlgorithm concept is not mandatory for algorithms but using it allows creation of nested algorithms (including runtime-switched algorithms) in unified manner using one utils/params construcor call.
If at least one algorithm in hierarchy is not HierarchicAlgorithm you are not able to create hierarchy this way (it will cause runtime error).
Rationale: why runtime error instead of compile time? Because this way you can still combine HierarchicAlgorithm classes with ones that are not. You are simply need to use manual method of algorithms creation.

What is the difference between utils and params?

While this division at first glance seems artificial it helps to seggregate user defined params (number of iterations, tolerance, type of algorithm, etc) from programming enviroment (some additional classes that can be used by algorithms - logger, profiler, vector space class). Parameters supposed to be lightweight and can be constructed from code (linsolver_params.max_iterations = 100;) or using json or another config file. Utils also supposed to be lightweight but carry some internal programming dependencies.

How to use/write HierarchicAlgorithm ?

Idea of HierarchicAlgorithm is that its universal constructor creates not only algorithms itself but also all its subalgorithms that are specified as its template arguments.

Lets start with case where you dont have any subalgorithms. This time you can use something like this:
```
class HierarchicAlgorithmName
{
public:
    struct params_hierarchy
    {
        params_hierarchy(const std::string &log_prefix = "", const std::string &log_name = "HierarchicAlgorithmName::")
        {
        }
        /// Some parameters - number of iterations tolerances etc
    };
    struct utils_hierarchy
    {
        /// Some utils - log, profiler, vector spaces, etc
    };

    HierarchicAlgorithmName(const utils_hierarchy &u, const params_hierarchy &p);
};
```

Consider next example where some IterativeSolver (jacobi method or something) has one subalgorithm - Preconditioner:

```
template<class Preconditioner>
class IterativeSolver
{
    using params_hierarchy = ...;
    using utils_hierarchy = ...;

    IterativeSolver(const utils_hierarchy &u, const params_hierarchy &p)
    {
        prec_ = ??;
    }

private:
    std::shared_ptr<Preconditioner> prec_;
};
```

Preconditioner here is subalgorithm. But how IterativeSolve can create instance of Preconditioner without knowing what kind of parameters it needs? If Preconditioner implements HierarchicAlgorithm convention we can do it this way:

```
template<class Preconditioner>
class IterativeSolver
{
    struct params_hierarchy 
    {
        typename Preconditioner::params_hierarchy prec;
        ...
    };
    using utils_hierarchy
    {
        typename Preconditioner::utils_hierarchy prec;
        ...
    };

    IterativeSolver(const utils_hierarchy &u, const params_hierarchy &p)
    {
        prec_ = std::make_shared<Preconditioner>(u.prec, p.prec);
    }

private:
    std::shared_ptr<Preconditioner> prec_;
};
```

However if one wants to use this IterativeSolver with Preconditioner that is not HierarchicAlgorithm it will cause compile time error. To overcome this restriction we can use internal nmfd helper classes and functions:

```
template<class Preconditioner>
class IterativeSolver
{
    struct params_hierarchy 
    {
        typename nmfd::detail::algo_params_hierarchy<Preconditioner>::type prec;
        ...
    };
    using utils_hierarchy
    {
        typename nmfd::detail::algo_utils_hierarchy<Preconditioner>::type prec;
        ...
    };

    IterativeSolver(const utils_hierarchy &u, const params_hierarchy &p)
    {
        prec_ = nmfd::detail::algo_hierarchy_creator<Preconditioner>::get(utils.prec,prm.prec);
    }

private:
    std::shared_ptr<Preconditioner> prec_;
};
```

algo_params_hierarchy/algo_utils_hierarchy checks whether its parameter has its own nested params_hierarchy/utils_hierarchy and if it has not - substitute it with some dummy structure. algo_hierarchy_creator checks whether its parameter is HierarchicAlgorithm (ie it has corresponding nested types and constructor). If it has - calls it, otherwise - throws an exception (why runtime check is used we explained earlier).

Recomendation - is to use additional params and utils nested types to separate own parameters from parameters of nested algorithms. Also some 'manual' constructors should be added to allow user set preconstructed subalgorithms:

```
template<class SubAlgo1,class SubAlgo2>
class HierarchicAlgorithmName
{
public:
    struct params
    {
        params(const std::string &log_prefix = "", const std::string &log_name = "HierarchicAlgorithmName::")
        {
        }
        ...
    };
    struct params_hierarchy : public params
    {
        typename nmfd::detail::algo_params_hierarchy<SubAlgo1>::type sub_algo1;
        typename nmfd::detail::algo_params_hierarchy<SubAlgo2>::type sub_algo2;

        TODO add constructor sample with prefixes formation
    };
    struct utils
    {
        ...
    };
    struct utils_hierarchy : public utils
    {
        typename nmfd::detail::algo_utils_hierarchy<SubAlgo1>::type sub_algo1;
        typename nmfd::detail::algo_utils_hierarchy<SubAlgo2>::type sub_algo2;

        TODO add constructor sample
    };

    /// Manual constructor
    HierarchicAlgorithmName(std::shared_ptr<SubAlgo1> sub_algo1, std::shared_ptr<SubAlgo1> sub_algo2, const params &u = utils(), const params &p = params());
    /// Automatic constructor
    HierarchicAlgorithmName(const utils_hierarchy &u, const params_hierarchy &p);
};
```

TODO

# Auxilary classes description

## Glued Vector Space

TODO (add class from Stokes, add description)

## Static Vector Space

TODO (implement class)

```
template<class T, int Dim>
struct static_vector_space
{
    using vector_type = std::array<T,Dim>;

    /// VectorSpace interface
};
```

## Pair Vector Space

TODO (implement class)

```
template<class VectorSpace1,class VectorSpace2>
struct pair_vector_space
{
    using vector1_type = typename VectorSpace1::vector_type;
    using vector2_type = typename VectorSpace2::vector_type;

    using vector_type = std::pair<vector1_type,vector2_type>;

    /// VectorSpace interface
};
```

## Dense Extended Operator

## Tuple Vector Space

TODO (no class, no description)
Do we need it at all??

## Rank1 Updated Operator

Is OrigOperator been OperatorWithSpaces?

```
template<class OrigOperator, class VectorSpace?>
struct rank1_updated_operator
{
    using vector_type = typename OrigOperator::vector_type;

    rank1_updated_operator(std::shared_ptr<const OrigOperator> orig_op = nullptr?);
    rank1_updated_operator(std::shared_ptr<const OrigOperator> orig_op, const vector_type &u, const vector_type &v);

    void set_operator(std::shared_ptr<const OrigOperator> orig_op);
    void set_rank1_vectors(const vector_type &u, const vector_type &v);

    const vector_type &u()const;
    const vector_type &v()const;

    /// OperatorWithSpaces interface?
};
```

## Sherman-Morrison Linear Solver For Rank1 Updated Systems

```
template<class OrigOperator, class VectorSpace?, class OrigSolver>
class sherman_morrison_rank1_updated_solver
{
public:
    using vector_type = ...;
    using operator_type = rank1_updated_operator<OrigOperator, VectorSpace?>;
public:
    void set_operator(std::shared_ptr<const operator_type> operator);
    bool solve(const vector_type &rhs, vector_type &res);
};
```

# Basic algorithms classes discription

## MG class template

mg is Preconditioner.

```
/// SystemOperator is OperatorWithSpaces.
/// Restrictor, Prolongator are Operator.
/// Smoother, CoarseSolver are Preconditioner
/// Coarsening is Coarsening
/// Smoother, CoarseSolver, Coarsening are HierarchicAlgorithm
/// All vector_space_type, vector_type, scalar_type are the same
template
<
    class SystemOperator,
    class Restrictor,
    class Prolongator,
    class Smoother,
    class CoarseSolver,
    class Coarsening
>
class mg
{
};
```

## TODO iterational linear solvers classes

## nonlinear_solver class template

```
template
<
    class VectorSpace, 
    class NonlinearOperator, 
    class IterationOperator, 
    class ConvergenceStrategy
>
class nonlinear_solver
{
};
```