#include <iostream>
    using std::cin;
    using std::cout;
    using std::endl;


int main() {
  int A, B;
  cin >> A >> B;
 
  // �����Ƀv���O������ǋL
  int i = 0;
  cout << "A:";
  while (i++ < A)
  {
    cout << "]"; // << endl; A�̖_�O���t
  }
  
  cout << endl << "B:";

  int j = 0;
  while (j++ < B)
  {
    cout << "]"; // << endl; B�̖_�O���t
  }
  cout << endl;
  
}