#include <bits/stdc++.h>
using namespace std;

int main() {
//   string str1, str2;
//   cin >> str1;
//   str2 = ", world!";

//   cout << str1 + str2 << endl; // ����ł������@str1 << str2
//   cout << (str1+str2).size() << endl; // �����𑪒�
//   cout << (str1+str2).at(0) << endl; // str�̐擪�ɃA�N�Z�X
//   char a = '*'; // �ꕶ���������Ƃ��� '�V���O���R�[�e�[�V����' �ň͂� ... ""�͕�����Ƃ��Ĉ�����
//   cout << a << endl;

//   string s, t;
//   getline(cin, s); // �ϐ�s�œ��͂���s�󂯎��
//   getline(cin, t); // �ϐ�t�œ��͂���s�󂯎��
 
//   cout << "��s�� " << s << endl;
//   cout << "��s�� " << t << endl;


  string S;
  cin >> S;

  // �����Ƀv���O������ǋL
  
  int i, result = 1;
  for(i = 0; i < S.size(); i++){
    if(S.at(i) == '+'){
        result += 1;
    }else if(S.at(i) == '-'){
        result -= 1;
    }
  }

  cout << result << endl;

}
