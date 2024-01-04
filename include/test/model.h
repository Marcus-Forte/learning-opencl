template <class Scalar>
struct Model {
  Model(Scalar *x, Scalar *y) : data_x(x), data_y(y) {}
  // API simply has to override this method

  bool f(const Scalar *x, Scalar *residual, unsigned int index) const {
    residual[0] =
        data_y[index] - (x[0] * data_x[index]) / (x[1] + data_x[index]);
    return true;
  }

 private:
  const Scalar *const data_x;
  const Scalar *const data_y;
};