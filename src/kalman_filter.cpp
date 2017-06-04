#include "kalman_filter.h"
#include "tools.h"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd y = z - h_x_();

  y(1) = constrainAngle(y(1));
  
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();
  
  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

VectorXd KalmanFilter::h_x_() {
  VectorXd ret = VectorXd(3);
  auto px = x_[0];
  auto py = x_[1];
  auto vx = x_[2];
  auto vy = x_[3];

  if (fabs(px) < 0.001) {
    cout << "Detected 0 division!" << endl;
    px = 0.001;
  }
  
  double phi = atan2(py, px);
  double rho = sqrt(px * px + py * py);
  
  // TODO: factor out a by-reference helper
  if (fabs(rho) < 0.001) {
    cout << "Detected 0 division!" << endl;
    rho = 0.001;
  }
  
  ret << rho,
         phi,
         (px * vx + py * vy) / rho;
  return ret;
}

double KalmanFilter::constrainAngle(double x) {
  x = fmod(x + M_PI, 2 * M_PI);
  if (x < 0) {
    x += 2 * M_PI;
  }
  return x - M_PI;
}
