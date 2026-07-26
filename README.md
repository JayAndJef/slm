# Jayden's Small Language Model

Current model is 185m params. Core model and tokenizer rolled by hand, but default tokenizer is huggingface's. Default attention is also flashattention/pytorch fused kernels. Trained on a trimmed version of cosmopedia.

### Example

When asked to complete `How do computers think?`, the model gives:

```
Well, they don't have brains like humans do. Instead, they use something called 'programming.' Imagine having a set of instructions that tell a computer exactly what to do. This is similar to how we write our own sentences!

One popular programming language used by scientists today is called C++. It's kind of like the classic programming language, but with some special tricks. In C++, instead of writing long commands like "print number 3," we write short ones like "print int(1)". These are called functions, and they let us do cool things like add two numbers together, subtract one number from another, or even divide one number by another!

Now, here's where it gets really interesting. Even though C++ is a different language than Python (which you might know as a language for math), it still uses a similar concept of being "typed" – meaning it's organized and structured so that it's easy to read and understand. That makes C++ a great choice for scientists who want to communicate their ideas clearly and efficiently.

So why do people care about programming languages like C++? Because they make it easier for us to create programs that perform complex calculations, analyze data, and simulate real-world phenomena. By learning these concepts early on, you'll be well-equipped to tackle any challenge that comes your way!
```