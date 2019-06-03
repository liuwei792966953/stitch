#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>


//using SparseMatrixd = Eigen::SparseMatrix<double>;
using SparseMatrixd = Eigen::SparseMatrix<double,Eigen::RowMajor>;

class Energy {
public:
	void get_reduction(std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights);

	void update(const SparseMatrixd &D, const Eigen::VectorXd &x, Eigen::VectorXd &z, Eigen::VectorXd &u);

	virtual int get_dim() const = 0;

	virtual double get_weight() const = 0;

	virtual Eigen::VectorXd reduce(const Eigen::VectorXd& x) const = 0;

protected:
	virtual void get_reduction(std::vector< Eigen::Triplet<double> > &triplets) = 0;

	virtual void prox(Eigen::VectorXd &zi) = 0;

private:
	int g_index;
};


class Lame {
public:
	static Lame rubber(){ return Lame(10000000,0.499); } // true rubber
	static Lame soft_rubber(){ return Lame(10000000,0.399); } // fun rubber!
	static Lame very_soft_rubber(){ return Lame(1000000,0.299); } // more funner!

	double mu, lambda;
	double bulk_modulus() const { return lambda + (2.0/3.0)*mu; }

	// Hard strain limiting (e.g. [0.95,1.05]), default no limit
	// with  min: -inf to 1, max: 1 to inf.
	// In practice if max>99 it's basically no limiting.
	double limit_min, limit_max;

	// k: Youngs (Pa), measure of stretch
	// v: Poisson, measure of incompressibility
	Lame( double k, double v ) :
		mu(k/(2.0*(1.0+v))),
		lambda(k*v/((1.0+v)*(1.0-2.0*v))),
		limit_min(-100.0),
		limit_max(100.0) {}

	// Use custom mu, lambda
	Lame(): limit_min(-100.0),
		limit_max(100.0) {}
};


class TriEnergy : public Energy {
protected:
	typedef Eigen::Matrix<int,3,1> Vec3i;
	typedef Eigen::Matrix<double,3,1> Vec3;
	typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;
	Vec3i tri;
	Lame lame;

	double area;
	double weight;
	Eigen::Matrix2d rest_pose;

	Eigen::Matrix<double,3,2> S_;

public:
	int get_dim() const { return 6; }
	double get_weight() const { return weight; }

	TriEnergy( const Vec3i &tri_, const std::vector<Vec3> &verts, const Lame &lame_ );

	void get_reduction( std::vector< Eigen::Triplet<double> > &triplets );
	

	// Unless derived from uses linear strain (no area conservation)
	virtual void prox( VecX &zi );

	virtual Eigen::VectorXd reduce(const Eigen::VectorXd& x) const;
};

/*
class SpringPin : public Energy {
protected:
	typedef Eigen::Matrix<int,3,1> Vec3i;
	typedef Eigen::Matrix<double,3,1> Vec3;
	typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;
	int idx; // constrained vertex
	Vec3 pin; // location of the pin
	bool active;
	double weight;

public:
	int get_dim() const { return 3; }
	double get_weight() const { return weight; }
	void set_pin( const Vec3 &p ){ pin = p; }
	void set_active( bool a ){ active = a; }

	SpringPin( int idx_, const Vec3 &pin_ ) : idx(idx_), pin(pin_), active(true) {
		// Because we usually use bulk mod of rubber for elastics,
		// We'll make a really strong rubber and use that for pin.
		Lame lame = Lame::rubber();
		weight = std::sqrt(lame.bulk_modulus()*2.0);
	}

	void get_reduction( std::vector< Eigen::Triplet<double> > &triplets ){
		const int col = 3*idx;
		triplets.emplace_back( 0, col+0, 1.0 );
		triplets.emplace_back( 1, col+1, 1.0 );
		triplets.emplace_back( 2, col+2, 1.0 );
	}

        Eigen::VectorXd reduce(const Eigen::VectorXd& x) const {
            return x.segment<3>(3*idx);
        }

	void prox( VecX &zi ){ if( active ){ zi = pin; } }
};
*/

