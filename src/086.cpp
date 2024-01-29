#include <iostream>

using namespace std;

int main(){

    int a, b;

    cin >> a >> b;

    int ans;
    ans = a * b;

    if(ans%2 == 1){
        cout << "Odd" << endl;
    }else{
        cout << "Even" << endl;
    }

    return 0;
    
}