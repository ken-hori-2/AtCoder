#include <iostream>
    using std::cin;
    using std::cout;
    using std::endl;


int main() {
  int A, B;
  cin >> A >> B;
 
  // ここにプログラムを追記
  int i = 0;
  cout << "A:";
  while (i++ < A)
  {
    cout << "]"; // << endl; Aの棒グラフ
  }
  
  cout << endl << "B:";

  int j = 0;
  while (j++ < B)
  {
    cout << "]"; // << endl; Bの棒グラフ
  }
  cout << endl;
  
}