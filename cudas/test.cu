#include<stdio.h>
#include<math.h>
#include<iostream>

int *a;
int *b;
int *c;

void vectorAdd(int *a,int*b,int*c,int n);

void vectorAdd(int *a,int *b,int *c,int n)
{
	for(int i=0;i<n;i++)
	{
		c[i] = a[i]+b[i];
	}
}

int main(){
	printf("\n Code to add vectors A and B");
	a = (int *)malloc(1024 * sizeof(int));
	b = (int *)malloc(1024*sizeof(int));
	c = (int *)malloc(1024*sizeof(int));

	/*Fill in the vectors:*/
	for(int i=0;i<1025;i++)
	{
		a[i]=i; b[i]=i; c[i]=0;
	}

	vectorAdd(a,b,c,1024);
	int size;
	size = (sizeof(c)/sizeof(c[0]));
	printf("\n first 10 elements of C is:");
		for(int i=0;i<10;i++){
			printf("\n%d",c[i]);
		}
	printf("\n freeing all vectors from memory");
	free(a); free(b); free(c);
	std::cout<<"Done";
}
