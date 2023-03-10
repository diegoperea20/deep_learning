/******************************************************************
 * ArduinoANN - An artificial neural network for the Arduino
 * All basic settings can be controlled via the Network Configuration
 * section.
 * See robotics.hobbizine.com/arduinoann.html for details.
 ******************************************************************/

#include <math.h>
#include <stdio.h>

/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/

const int PatternCount = 1;
const int HiddenNodes = 10;
const int InputNodes = 8;
const int OutputNodes = 1;

//DATA, correspondiente a 10 datos del Google Colab
const float x[8][8] = {
  //    HOUR   TEMPERATURE  HUMIDITY  WIND SPEED VISIBI   DEW P. TEM  SOLAR RAD  SEASONS
  { -0.304348,  0.573427,  0.408163, -0.513514, 0.998986,  0.785467, -0.602273, 0.333333},  // 0
  {  0.043478,  0.762238,  0.040816, -0.432432,-0.216422,  0.788927,  0.823864, 0.333333},  // 1
  {  0.217391,  0.811189,  0.020408, -0.675676, 0.740497,  0.823529, -0.045455, 0.333333},  // 2
  {  0.565217,  0.213287, -0.040816, -0.621622, 0.632032,  0.249135, -0.937500, 1.000000},  // 3
  { -0.391304, -0.153846,  0.040816, -0.729730, 0.389762, -0.044983, -0.892045,-0.333333},  // 4
  { -0.913043,  0.601399,  0.448980, -0.837838, 0.796249,  0.826990, -1.000000, 0.333333},  // 5
  {  0.739130,  0.716783,  0.326531, -0.459459, 0.580335,  0.885813, -0.988636, 0.333333},  // 6
  {  0.913043, -0.304196, -0.163265, -0.729730, 0.857070, -0.280277, -1.000000, 1.000000},  // 7
};

const float b[8][1] = {
  //    salida Colab
  { -0.46319914 },  // 0
  { -0.40030032 },  // 1
  { -0.36325067 },  // 2
  { -0.29922786 },  // 3
  { -0.79215720 },  // 4
  { -0.63503003 },  // 5
  {  0.05618465 },  // 6
  { -0.46307456 }   // 7
};

// Pesos capa oculta
const float HiddenWeights[HiddenNodes][InputNodes+1]= {
    {    6.8647270e-05,   -1.3720945e+00,   4.9748176e-01,   -1.9836724e-01,   4.7617614e-02,  -5.8109200e-01,  -4.3505394e-01,   3.4052521e-02,   0.51824534},
    {    5.6620789e-01,    5.0238663e-01,  -7.4036109e-01,    2.9759112e-01,   6.4913101e-02,   4.4995901e-01,  -1.7834599e-01,   5.6578744e-02,  -0.23224986},
    {   -1.0353112e-01,   -4.2780468e-01,   3.1550148e-01,    3.4774268e-01,   1.2590386e-02,  -6.4925835e-02,   2.6990864e-01,   2.2391453e-01,  -0.37652773},
    {   -4.9398108e-03,   -4.7172624e-01,  -1.0887721e-01,   -2.9813299e-01,   3.6151993e-01,  -1.6322175e-01,   1.5972377e-01,   9.1011599e-02,   0.158625},
    {   -2.8814337e-01,   -1.4869051e-01,  -3.9423314e-01,    5.5221564e-01,  -4.6224520e-01,   3.9156488e-01,   5.2121747e-01,   6.1398387e-02,  -0.44157675},
    {   -3.6311933e-01,   -1.1599174e+00,   2.2722551e-01,   -1.2367419e-01,  -5.8868214e-02,  -3.1407550e-01,  -2.6539102e-01,   3.7389579e-01,   0.08572464},
    {    5.0457025e-01,   -2.0985128e-01,  -4.8565936e-01,   -3.8402212e-01,   2.4121650e-02,  -8.5143608e-01,  -3.2076508e-01,  -1.9487152e-01,   0.9303143},
    {    1.6162302e-01,    2.3086137e-01,   3.0849808e-01,   -3.2091805e-01,  -3.2921389e-02,  -4.2425239e-01,   6.2542301e-01,   1.8506920e-01,  -0.4012757},
    {   -1.6705030e-01,   -3.0139875e-01,  -1.1266586e+00,   -5.1539767e-01,   2.6786050e-01,  -3.4564421e-01,   1.4123514e-01,  -2.9643688e-02,   0.5335912},
    {   -4.4064024e-01,   -3.6237752e-01,   1.9358855e-01,   -1.1888021e-01,   1.4336103e-01,   4.3869618e-02,   5.0580129e-03,  -3.6023954e-01,  -0.2633135}

};

// Pesos capa de salida
const float OutputWeights[OutputNodes][HiddenNodes+1]  = {
{-0.82207495,   0.3115925,   0.29827258,   0.46822974,  1.2734339,    0.53441507,   0.5313172,   0.8889937,    -0.50083876,   0.24386558, -0.5555744}
};  


/******************************************************************
 * End Network Configuration
 ******************************************************************/


int i, j, p, q, r, k,o;
float Accum;
float Hidden[HiddenNodes];
float Output[OutputNodes];
float Input[InputNodes][PatternCount];
float Colab;



void setup(){
  //start serial connection
  Serial.begin(9600);
  randomSeed(analogRead(0));

}

void loop(){
  //Recorre las diferentes columnas para rellenar el vector Input con cada uno de los atributos
  for( o = 0 ; o < 8 ; o++ ) {
    Input[o][0] = x[k][o];
    }

   Colab = b[k][0];
/******************************************************************
* Compute hidden layer activations
******************************************************************/

    for( i = 0 ; i < HiddenNodes ; i++ ) {    
      Accum = HiddenWeights[i][InputNodes] ;
      for( j = 0 ; j < InputNodes ; j++ ) {
        Accum += HiddenWeights[i][j]*Input[j][0];
      }
      if (Accum < 0) {
        Hidden[i] = 0; 
      }
      else {
        Hidden[i] = Accum;
      }
    }
    
/******************************************************************
* Compute output layer activations
******************************************************************/
 
    for( i = 0 ; i < OutputNodes ; i++ ) {    
      Accum = OutputWeights[i][HiddenNodes] ;
      for( j = 0 ; j < HiddenNodes ; j++ ) {
        Accum +=  OutputWeights[i][j]*Hidden[j];
      }
      Output[i] = Accum; 
    }

    //Contador para acceder a cada una de las filas en cada ciclo
    k++;
    if (k == 8) {
      k=0;
    }

    //Enseña en el serial los valores de las predicciones
    Serial.print("OutputTest_Arduino:");
    Serial.print(Output[0]);
    Serial.print(",");
    Serial.print("OutputTest_Colab:");
    Serial.print(Colab);
    Serial.print("\n");
    
    delay(1000);
}
