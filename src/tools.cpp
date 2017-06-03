#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() == 0) {
    throw invalid_argument("received 0 length vector");
  }
  if (estimations.size() != ground_truth.size()) {
    throw invalid_argument("estimations and ground_truth does not match");
  }
  
  
  //accumulate squared residuals
  for(int i=0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array()*residual.array();
    rmse += residual;
  }
  //calculate the mean
  rmse = rmse / estimations.size();
  return rmse.array().sqrt();
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  
  float pow2_px_py = pow(px, 2) + pow(py, 2);
  float sqr_px_py = sqrt(pow2_px_py);
  float pow2_px_py_15 = pow(pow2_px_py, 1.5);
  
  //check division by zero
  if (fabs(sqr_px_py) < 0.0001 || fabs(pow2_px_py) < 0.0001 || fabs(pow2_px_py_15) < 0.0001) {
    cout << "Detected 0 division!" << endl;
    return Hj;
  }
  
  //compute the Jacobian matrix
  Hj << px / sqr_px_py, py / sqr_px_py, 0, 0,
        -py / pow2_px_py, px / pow2_px_py, 0, 0,
        (py * (vx * py - vy * px)) / pow2_px_py_15,
        (px * (vy * px - vx * py)) / pow2_px_py_15,
        px / sqr_px_py, py / sqr_px_py;
  
  return Hj;
}
