# triton-linalg

A shared middle-layer for the Triton Compiler. Its approach is similar to
Microsoft's [triton-shared](https://github.com/microsoft/triton-shared.git),
but there are differences in some pointer handling.
Currently, it has successfully supported the Cambrian backend as a front-end representation,
and functionally, it is capable of handling nearly all features of the Triton language.
