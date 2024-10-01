#include <iostream>
using namespace std;

int main()
{
    // Assing type to multiple variables individually
    int x = 5;
    int y = 6;
    int z = x + y;

    // Assing single type to multiple variables at once
    int a = 2, b = 3, c = 2;
    int d = a + b + c;

    // Multiple Vaiables Type
    int integer_variable = 10;
    double double_variable = 10.5;
    char char_variable = 'A';
    string string_variable = "Hello, World!";
    bool bool_variable = true;

    // Print the sum of x and y
    cout << "The sum of x and y is: " << z << endl;

    cout << "The sum of a, b and c is: " << d << endl;

    cout << string_variable << "," << bool_variable << " I have " << integer_variable << " candies or " << double_variable << " candies " << char_variable << " Good Question!" << endl;

    return 0;
}