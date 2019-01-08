#include <stdio.h>      /* printf, NULL */
void func1(int* u);
void func3(){

}
void func2(){

}
void func1(int* u){
	printf("Soulja Boi\n");
	printf("me is %d\n",u[0]);
}

void main(){
	printf("HELLO\n");
	int *me = ((int*) malloc(10*sizeof(int)));
	me[0] = 1;

	func1(me);
}