使用IWYU报错后有如下一些原则：

- 首先我们对原生inc进行了一定的调整，所有的.cpp.inc文件都会被视为keep（即只要include了就要保留），所有的.h.inc文件都视为export（即只要include了都只需要include它的上级文件即代表选择了它的符号），但依然有一些行为需要使用宏https://github.com/include-what-you-use/include-what-you-use/blob/master/docs/IWYUPragmas.md 进行调整，包括并不限于：
  1. 使用.h来包括生成的inc文件时，对.h的分析目前无法将inc与h视为一个整体，所以.h上面为了使inc合法的前置声明需要使用keep pragma来保留；使用.cpp来包括生成的inc文件也是同理。
  2. 在能使用前置声明时（只使用了不需要完整类型的部分），iwyu会建议将include改成前置声明减少依赖，此部分分析结果有时会导致循环报错（譬如include了.h后iwyu希望改成前置声明，改成前置声明后它又需要使用.h），此时选择一个方案固定并改成keep。
  3. 一些隐含很深的依赖可能iwyu@clang15无法看出来（目前已知的有uniqueptr需要llvm的内置注册器配合，而注册器需要Dialect的完整类型，此时Dialect的前置声明会无法编译通过），也需要代入头文件并改成keep。
  4. iwyu在给出建议的时候会建议includeinc文件，不要听，使用它所在的.h文件，inc已经被标定为export，所以总是可以通过检查。
  5. 如果IWYU建议你用<>include mlir/LLVM头文件，不用管他，使用\"\"。
  6. 调整代码后需要在本地重新执行IWYU迭代，因为头文件修改造成的符号变更可能使IWYU调整它的结论。
