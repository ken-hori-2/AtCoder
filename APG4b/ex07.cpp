// #include <bits/stdc++.h>
// using namespace std;

#include <iostream>
  using std::cin;
  using std::cout;
  using std::endl;
  using std::string;

int main() {
  // �ϐ�a,b,c��true�܂���false��������AtCoder�Əo�͂����悤�ɂ���B
  bool a = true; // true �܂��� false
  bool b = false; // true �܂��� false
  bool c = true; // true �܂��� false

  // ���������͕ύX���Ȃ�����

  if (a) { // a = True
    cout << "At";
  }
  else {
    cout << "Yo";
  }

  if (!a && b) { // a = False, b  = True
    cout << "Bo";
  }
  else if (!b || c) { // b = False, c = True
    cout << "Co";
  }

  if (a && b && c) {
    cout << "foo!";
  }
  else if (true && false) {
    cout << "yeah!";
  }
  else if (!a || c) { // a = False, c = True
    cout << "der";
  }

  cout << endl;
}
