#include "energy.hpp"

#include <Eigen/SVD>


void Energy::get_reduction( std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights ){
	std::vector< Eigen::Triplet<double> > temp_triplets;
	get_reduction( temp_triplets );
	int n_trips = temp_triplets.size();
	g_index = weights.size();
	for( int i=0; i<n_trips; ++i ){
		const Eigen::Triplet<double> &trip = temp_triplets[i];
		triplets.emplace_back( trip.row()+g_index, trip.col(), trip.value() );
	}
	int dim = get_dim();
	double w = get_weight();
	if( w <= 0.0 ){
		throw std::runtime_error("**Energy::get_reduction Error: Some weight leq 0");
	}
	for( int i=0; i<dim; ++i ){ weights.emplace_back( w ); }
}

void Energy::update(const SparseMatrixd &D, const Eigen::VectorXd &x, Eigen::VectorXd &z, Eigen::VectorXd &u){
	int dof = x.rows();
	int dim = get_dim();

	//Eigen::VectorXd Dix = D.block(g_index,0,dim,dof) * x;
	Eigen::VectorXd Dix = reduce(x);
	Eigen::VectorXd zi = Dix + u.segment(g_index,dim);
	prox(zi);
	u.segment(g_index,dim).noalias() += Dix - zi;
	z.segment(g_index,dim).noalias() = zi;

        //Eigen::VectorXd Dix = D.block(g_index,0,dim,dof)*x;
        //Eigen::VectorXd ui = u.segment(g_index,dim);
        //Eigen::VectorXd zi = Dix + ui;
	//prox( zi );
	//ui += (Dix - zi);
	//u.segment(g_index,dim) = ui;
	//z.segment(g_index,dim) = zi;
}


TriEnergy::TriEnergy( const Vec3i &tri_, const std::vector<Vec3> &verts, const Lame &lame_ ) :
	tri(tri_), lame(lame_), area(0.0), weight(0.0) {

	if( lame.limit_min > 1.0 ){ throw std::runtime_error("**TriEnergy Error: Strain limit min should be -inf to 1"); }
	if( lame.limit_max < 1.0 ){ throw std::runtime_error("**TriEnergy Error: Strain limit max should be 1 to inf"); }
	Vec3 e12 = verts[1] - verts[0];
	Vec3 e13 = verts[2] - verts[0];
	Vec3 n1 = e12.normalized();
	Vec3 n2 = (e13 - e13.dot(n1)*n1).normalized();
	Eigen::Matrix<double,3,2> basis;
	Eigen::Matrix<double,3,2> edges;
	basis.col(0) = n1; basis.col(1) = n2;
	edges.col(0) = e12; edges.col(1) = e13;

        Eigen::Matrix2d F = basis.transpose() * edges;
	rest_pose = F.inverse(); // Rest pose matrix

	area = F.determinant() / 2.0f;
	if( area < 0.0 ){
		throw std::runtime_error("**TriEnergy Error: Inverted initial pose");
	}

	double k = lame.bulk_modulus();
	weight = std::sqrt(k*area);

	S_.setZero();
	S_(0,0) = -1;	S_(0,1) = -1;
	S_(1,0) =  1;   S_(2,1) =  1;
}


void TriEnergy::get_reduction( std::vector< Eigen::Triplet<double> > &triplets ){

	Eigen::Matrix<double,3,2> D = S_ * rest_pose;
	int cols[3] = { 3*tri[0], 3*tri[1], 3*tri[2] };
	for( int i=0; i<3; ++i ){
		for( int j=0; j<3; ++j ){
			triplets.emplace_back( i, cols[j]+i, D(j,0) );
			triplets.emplace_back( 3+i, cols[j]+i, D(j,1) );
		}
	}
}

Eigen::VectorXd TriEnergy::reduce(const Eigen::VectorXd& x) const {
	const Eigen::Matrix<double,3,2> D = S_ * rest_pose;
	const int cols[3] = { 3*tri[0], 3*tri[1], 3*tri[2] };

	Eigen::Matrix<double,6,1> z = Eigen::Matrix<double,6,1>::Zero();
	for (int i=0; i<3; ++i ) {
	    for (int j=0; j<3; j++) {
	        z[i]   += D(j,0) * x[cols[j]+i];
	        z[3+i] += D(j,1) * x[cols[j]+i];
            }
	}

	return z;
}

void TriEnergy::prox(VecX &zi) {
	using namespace Eigen;
	typedef Matrix<double,6,1> Vector6d;

	JacobiSVD<Matrix<double,3,2>> svd(Map<Matrix<double,3,2> >(zi.data()), ComputeThinU | ComputeThinV);
	Matrix<double,3,2> P = svd.matrixU().leftCols(2) * svd.matrixV().transpose();
	zi = 0.5 * ( Map<Vector6d>(P.data()) + zi );

	const bool check_strain = lame.limit_min > 0.0 || lame.limit_max < 99.0;
	if( check_strain ){
		double l_col0 = zi.head<3>().norm();
		double l_col1 = zi.tail<3>().norm();
		if( l_col0 < lame.limit_min ){ zi.head<3>() *= ( lame.limit_min / l_col0 ); }
		if( l_col1 < lame.limit_min ){ zi.tail<3>() *= ( lame.limit_min / l_col1 ); }
		if( l_col0 > lame.limit_max ){ zi.head<3>() *= ( lame.limit_max / l_col0 ); }
		if( l_col1 > lame.limit_max ){ zi.tail<3>() *= ( lame.limit_max / l_col1 ); }
	}
}


