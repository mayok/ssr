#include <stdio.h>
#include <vector>
#include <cmath>
#include <complex>
#include <map>
#include <algorithm>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <emscripten/emscripten.h>

#define K 6
#define D 12
#define PI 4*atan(1.0)
#define NUM_FFT 2048

extern "C" {

  /*
   * MFCC
   *
   */
  // fft
  EMSCRIPTEN_KEEPALIVE
  void fft(std::vector<float>& _signal, std::vector<float>& im) {
    // std::vector<std::complex<float> > でできる
    size_t n = _signal.size();

    for(int i = 0; i < n; i++) {
      int t = 0;
      for(int j = 0, h = i, k = n; ; h >>= 1) {
        k>>=1;
        if(k == 0) {
          t = j;
          break;
        }

        j = (j << 1) | (h & 1);
      }

      if(t > i) {
        std::swap(_signal[i], _signal[t]);
      }
    }
    for(int hn = 1; hn * 2 <= n; hn *= 2) {
      for(int i = 0; i < n; i+= hn * 2) {
        for(int j = i; j < i + hn; j++) {
          float _cos = cos(PI * (j - i) / hn);
          float _sin = sin(PI * (j - i) / hn);
          float tre = _signal[j+hn] * _cos + im[j+hn] * _sin;
          float tim = -1 * _signal[j+hn] * _sin + im[j+hn] * _cos;

          _signal[j+hn] = _signal[j] - tre;
          im[j+hn] = im[j] - tim;
          _signal[j] += tre;
          im[j] += tim;
        }
      }
    }
  }

 /*
  * mel filter bank
  *
  */
  EMSCRIPTEN_KEEPALIVE
  void melFilterBank(float filterbank[20][1024]) {
    int indexcenters[20] = { 6, 13, 21, 31, 42, 55, 71, 90, 112, 138, 169, 205, 248, 299, 358, 429, 512, 610, 726, 863 };
    int indexstart[20] = { 0, 6, 13, 21, 31, 42, 55, 71, 90, 112, 138, 169, 205, 248, 299, 358, 429, 512, 610, 726 };
    int indexstop[20] = { 13, 21, 31, 42, 55, 71, 90, 112, 138, 169, 205, 248, 299, 358, 429, 512, 610, 726, 863, 1024 };

    for(int i = 0; i < 20; i++) {

      float increment = 1.0 / ( indexcenters[i] - indexstart[i] );
      for(int j = indexstart[i]; j < indexcenters[i]; j++)
        filterbank[i][j] = ( j - indexstart[i] ) * increment;

      float decrement = 1.0 / ( indexstop[i] - indexcenters[i]);
      for(int j = indexcenters[i]; j < indexstop[i]; j++)
        filterbank[i][j] = 1.0 - (j - indexcenters[i]) * decrement;
    }
  }

  /*
  * pre emphasis filter and hamming window
  *
  */
  EMSCRIPTEN_KEEPALIVE
  void preEmphHamming(std::vector<float>& _signal) {
    size_t len = _signal.size();
    std::vector<float> y(len, _signal[0] * 0.08);

    for (int i = 1; i < len; i++) {
      y[i] = _signal[i] - ( 0.97 * _signal[i-1]);
      y[i] *= 0.54 - 0.46 * cos( 2.0 * PI * i / (len - 1) );
    }
    _signal = y;
  }

  /*
   *  fft, power spectrum
   *
   */
  EMSCRIPTEN_KEEPALIVE
  void powerSpectrum(std::vector<float>& _signal) {
    _signal.resize(NUM_FFT, 0);
    std::vector<float> im(_signal.size(), 0);
    fft(_signal, im);

   // power spectrum
   for(int i = 0; i<NUM_FFT/2+1; i++) {
    _signal[i] = _signal[i] * _signal[i] + im[i] * im[i];

    // XXX: we use amplitude spectrum
    _signal[i] = sqrt(_signal[i]);
   }
  }

  EMSCRIPTEN_KEEPALIVE
  void lmfb(std::vector<float>& _signal, std::vector<float>& mspec) {
    float filterbank[20][1024];
    melFilterBank(filterbank);

    for(int i = 0; i < 20; i++) {
      float t = 0.0;
      for(int j = 0; j < 1024; j++) {
        t += _signal[j] * filterbank[i][j];
      }
      mspec[i] = log10( t );
    }
  }

  /*
   * DCT: Discrete Cosine Transform
   *
   *            N-1
   *  y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
   *            n=0
   */
  EMSCRIPTEN_KEEPALIVE
  void dct(std::vector<float>& mspec) {
    std::vector<float> y(20);
    for(int k = 0; k < 20; k++) {
      float s = 0.0;
      for(int n = 0; n < 20; n++) {
        s += mspec[n] * cos( PI * k * ( 2 * n + 1 ) / 40.0);
      }
      if(k == 0)
        y[k] = sqrt(0.0125) * 2 * s;
      else
        y[k] = sqrt(0.025) * 2 * s;
    }
    mspec = y;
  }


  /*
   * GMM
   *
   */
  EMSCRIPTEN_KEEPALIVE
  float gaussian_a( Eigen::Matrix<float,D,1>& x ) {
    std::vector<Eigen::Matrix<float,D,1> >mu(K);
    std::vector<Eigen::Matrix<float,D,1> >tmp(K);
    std::vector<Eigen::Matrix<float,D,D> >sigma(K);

    mu[ 0 ] << 1.045289,-1.305828,-1.632999,-0.79128,0.735862,-0.75539,-1.005555,0.182137,-1.391732,0.461666,-0.508204,-1.812204 ;
    mu[ 1 ] << -1.022868,1.2255,0.065915,0.1007,-0.125436,-0.470051,1.042649,-0.477242,-0.175671,1.397663,-1.42869,0.715158 ;
    mu[ 2 ] << -1.321481,1.166594,0.881629,1.045384,-1.236299,0.96017,0.56784,-0.104132,1.201355,-0.381315,0.713802,0.606426 ;
    mu[ 3 ] << -0.266711,-0.486543,0.254529,-1.192058,0.008725,-0.797688,0.815909,-1.474869,-0.617526,0.89267,-1.263781,0.567462 ;
    mu[ 4 ] << 0.077669,0.821815,0.477573,1.700981,-0.836663,2.008036,0.089543,0.49323,0.806698,-0.601446,1.649284,-0.407271 ;
    mu[ 5 ] << 0.973848,-0.334144,0.016351,-0.008301,0.918384,-0.179857,-0.708653,1.069684,0.147832,-0.833491,0.560039,0.250159 ;

    tmp[ 0 ] << 0.027609,0.143805,0.142046,0.038757,0.022779,0.340125,0.839634,0.042827,0.106046,1.597478,0.40734,0.355934 ;
    tmp[ 1 ] << 0.002892,0.021577,0.024623,0.01192,0.034727,0.026705,0.036885,0.006351,0.008255,0.041397,0.019284,0.014053 ;
    tmp[ 2 ] << 0.048139,0.235065,0.232711,0.104258,0.282556,0.338484,0.630914,0.834265,0.129288,0.347942,0.204718,0.108062 ;
    tmp[ 3 ] << 0.002775,0.117054,0.321703,0.314324,0.119961,0.243523,0.226855,0.006881,0.234243,0.055041,0.058296,0.170705 ;
    tmp[ 4 ] << 0.003944,0.044854,0.081483,0.049739,0.035319,0.035782,0.040214,0.02186,0.061613,0.017297,0.011334,0.01308 ;
    tmp[ 5 ] << 0.096776,0.079347,0.508704,0.261405,0.635902,0.285404,0.193894,0.192263,0.223821,0.221591,0.033359,0.185941 ;
    for(int i=0;i<K;i++)
      sigma[i] = tmp[i].asDiagonal();

    float s = 0.0;
    for(int k=0; k<K; k++) {
      s += exp( -0.5 * ((x - mu[k]).transpose()).dot( sigma[k].inverse() * (x - mu[k])) )
        / pow(sqrt(2 * PI), D) * sqrt(sigma[k].determinant());
    }

    return s;
  }

  EMSCRIPTEN_KEEPALIVE
  float gaussian_i( Eigen::Matrix<float,D,1>& x ) {
    std::vector<Eigen::Matrix<float,D,1> >mu(K);
    std::vector<Eigen::Matrix<float,D,1> >tmp(K);
    std::vector<Eigen::Matrix<float,D,D> >sigma(K);

    mu[ 0 ] << -0.948244,0.147369,0.736382,0.176651,-0.374323,1.384001,0.214249,-0.597159,0.852908,-0.26386,0.696364,0.414395 ;
    mu[ 1 ] << 0.535614,0.906623,0.33812,1.330892,-0.723982,0.724014,0.224763,-0.504865,1.310725,-1.229114,0.760942,-0.870861 ;
    mu[ 2 ] << 1.427093,-1.22838,-1.068769,-0.169037,-0.669996,-1.103804,-1.298686,-0.28499,-0.799948,0.836463,-0.553465,-0.340691 ;
    mu[ 3 ] << -0.111131,0.653984,-0.701347,0.291937,0.808648,-0.087323,0.909245,-0.263152,-0.124629,-0.340603,-0.2224,0.761196 ;
    mu[ 4 ] << -0.851706,-0.463016,0.674259,-1.587806,0.803292,-0.834229,-0.361572,1.685811,-1.120796,1.462439,-0.844003,0.394169 ;
    mu[ 5 ] << 1.383722,-1.93198,-0.302948,-0.588463,-1.01017,-1.264276,-1.272249,-0.128749,-1.029568,-0.219256,0.156961,-2.177284 ;

    tmp[ 0 ] << 0.062961,0.3265,1.445208,0.195335,0.307244,0.083605,0.206115,0.218336,0.028302,0.047564,0.670093,0.579784 ;
    tmp[ 1 ] << 0.100823,0.061889,0.269028,0.086296,0.815075,0.371087,0.778233,0.561546,0.057075,0.141462,2.171398,0.643479 ;
    tmp[ 2 ] << 0.012896,0.378946,0.016237,0.136074,0.447804,0.040318,0.479484,0.095967,0.022333,0.020871,0.058113,0.099627 ;
    tmp[ 3 ] << 0.905271,0.443938,0.626571,0.218463,0.64765,0.194719,0.350059,0.163643,0.289874,0.218448,0.254773,0.064966 ;
    tmp[ 4 ] << 0.029748,0.220703,0.210212,0.097711,0.227356,0.093493,0.519068,0.664536,0.340536,0.481393,0.048106,0.431424 ;
    tmp[ 5 ] << 0.000625,0.003761,0.009922,0.001696,0.002349,0.023652,0.150298,0.004304,0.011944,0.005327,0.032983,0.030321 ;

    for(int i=0;i<K;i++)
      sigma[i] = tmp[i].asDiagonal();


    float s = 0.0;
    for(int k=0; k<K; k++) {
      s += exp( -0.5 * ((x - mu[k]).transpose()).dot( sigma[k].inverse() * (x - mu[k])) )
        / pow(sqrt(2 * PI), D) * sqrt(sigma[k].determinant());
    }

    return s;

  }

  EMSCRIPTEN_KEEPALIVE
  int gmm(std::vector<float> mspec) {

    Eigen::Matrix<float,D,1> x(mspec.data());
    float sa = 0.0,
          si = 0.0;

    sa = gaussian_a(x);
    si = gaussian_i(x);

    printf("%8.3e, %8.3e\n", sa, si);

    // 97: A, 105: I
    if(sa > si) return 97;
    return 105;
  }


  /*
   * main
   *
   */

  int main() {
    printf("Hello world!\n");
    return 0;
  }

  EMSCRIPTEN_KEEPALIVE
  int speechRecognition(float * _signal, size_t length) {
    std::vector<float> mspec(20);
    std::vector<float> s(length);
    for(int i=0; i<length; i++)
      s[i] = _signal[i];
    int v;

    preEmphHamming(s);
    powerSpectrum(s);
    lmfb(s, mspec);
    dct(mspec);

    printf("mfcc: \n");
    for(int i = 0; i < 12; i++) {
      printf("%d: %f\n", i, mspec[i]);
    }

    v = gmm(mspec);

    return v;

  }
}
