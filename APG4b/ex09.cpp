#include <iostream>

    using std::cin;
    using std::cout;
    using std::endl;


int main(){
    int x, a, b;

    cin >> x >> a >> b;
 
    // 1.�̏o��
    x++;
    cout << x << endl;
    
    // �����Ƀv���O������ǋL

    x *= (a+b);
    cout << x << endl; // 2.�̏o��

    x *= x;
    cout << x << endl; // 3.�̏o��

    x--;
    cout << x << endl; // 4.�̏o��
}