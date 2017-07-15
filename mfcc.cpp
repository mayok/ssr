// #include <stdio.h>
#include <cmath>
#include <vector>
#include <complex>
#include <map>
#include <algorithm>
#include <emscripten/emscripten.h>

#define PI 4*atan(1.0)
#define NUM_FFT 2048

extern "C" {
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
}
