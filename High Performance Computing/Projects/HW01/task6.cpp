#include <iostream>

using namespace std;

int main(int argc, char* argv[]) {
  // Check if the user has provided a command line argument
  if (argc != 2) {
    cout << "Please provide a command line argument N." << endl;
    return 1;
  }

  // Get the command line argument N
  int N = atoi(argv[1]);

  // Print the integers from 0 to N in ascending order
  for (int i = 0; i <= N; i++) {
    printf("%d ", i);
  }
  cout << endl;

  // Print the integers from N to 0 in descending order
  for (int i = N; i >= 0; i--) {
    cout << i << " ";
  }
  cout << endl;

  return 0;
}
