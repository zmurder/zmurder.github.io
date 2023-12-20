# 1 简单分类

## 1. 行内公式

将公式插入到本行内，符号：`$公式内容$`，如：`$xyz$`

![math](Markdown数学公式语法/math.svg)

## 2. 行间公式

将公式插入到新的一行内，并且居中，符号：`$公式内容$`，如：`$$xyz$$`
![xyz](Markdown数学公式语法/math.svg)

# 2 上标、下标与组合

1.  上标符号，符号：`^`，如：`$x^4$` ![x^4](Markdown数学公式语法/math-170115162522673.svg)

2.  下标符号，符号：`_`，如：`$x_1$` ![x_1](Markdown数学公式语法/math-17011515156962.svg)

3.  组合符号，符号：`{}`，如：![<sup>{16}_{8}O</sup>{2+}_{2}](Markdown数学公式语法/math-17011515156963.svg)

默认情况下，上、下标符号仅仅对下一个组起作用。一个组即单个字符或者使用{}（大括号） 包裹起来的内容。如果使用`$10^10$`表示的是![10^10](Markdown数学公式语法/math-17011515156964.svg),而`$10^{10}$` 才可以表示为![10^{10}](Markdown数学公式语法/math-17011515156965.svg)。同时，大括号还能消除二义性，如：`$x^5^6$` 将得到一个错误，必须使用大括号来界定^的结合性，如:`${x^5}^6$`表示的![{x^5}^6](Markdown数学公式语法/math-17011515156966.svg)：或者用`$x^{5^6}$`表示的![x^{5^6}](Markdown数学公式语法/math-17011515156977.svg)。

# 3 括号

## 3.1 小括号与方括号

用原始的( ) ，[ ] 即可，如`(2+3)[4+4]`可表示：![(2+3)[4+4]](Markdown数学公式语法/math-17011515156978.svg)。
 使用\left(或\right)使符号大小与邻近的公式相适应（该语句适用于所有括号类型），如\left(\frac{x}{y}\right)可表示![\left(\frac{x}{y}\right)](Markdown数学公式语法/right.svg)

## 3.2. 大括号

由于大括号{} 被用于分组，因此需要使用{和}表示大括号，也可以使用\lbrace 和\rbrace来表示。如{ab}或\lbrace ab\rbrace表示:![\{ab\}](Markdown数学公式语法/}.svg)

## 3.3. 尖括号

区分于小于号和大于号，使用\langle 和\rangle 表示左尖括号和右尖括号。如\langle x \rangle表示：![\langle x \rangle](Markdown数学公式语法/rangle.svg)

# 4 取整

## 4.1. 上取整

使用\lceil 和 \rceil 表示。 如，\lceil x \rceil表示为：![\lceil x \rceil](Markdown数学公式语法/rceil.svg)

## 4.2. 下取整

使用\lfloor 和 \rfloor 表示。如，\lfloor x \rfloor表示为:![\lfloor x \rfloor](Markdown数学公式语法/rfloor.svg)

# 5 求和\积分\连乘

## 5.1.求和

`\sum` 用来表示求和符号，其下标表示求和下限，上标表示上限。如:
 `$\sum_{r=1}^n$`表示：![\sum_{r=1}^n](Markdown数学公式语法/sum_{r%3D1}^n.svg)

## 5.2. 积分

`\int` 用来表示积分符号，同样地，其上下标表示积分的上下限。如，`$\int_{r=1}^\infty$`表示:![\int_{r=1}^\infty](Markdown数学公式语法/infty.svg)
 多重积分同样使用`\int` ，通过 i 的数量表示积分导数：
 如：
 `$\iint$` 表示为：![\iint](Markdown数学公式语法/iint.svg)
 `$\iiint$` 表示为：![\iiint](Markdown数学公式语法/iiint.svg)

## 5.3. 连乘

`$\prod {a+b}$` 表示：![\prod {a+b}](Markdown数学公式语法/prod {a%2Bb}.svg)
 `$\prod_{i=1}^{K}$` 表示：![\prod_{i=1}^{K}](Markdown数学公式语法/prod_{i%3D1}^{K}.svg)
 `$$\prod_{i=1}^{K}$$`表示（注意是行间公式）：![\prod_{i=1}^{K}](https://math.jianshu.com/math?formula=%5Cprod_%7Bi%3D1%7D%5E%7BK%7D)

## 5.4. 其他

与此类似的符号还有，
 `$\prod$` ：![\prod](Markdown数学公式语法/prod.svg)
 `$\bigcup$`：![\bigcup](https://math.jianshu.com/math?formula=%5Cbigcup)
 `$\bigcap$` ：![\bigcap](https://math.jianshu.com/math?formula=%5Cbigcap)
 `$arg\,\max_{c_k}$`：![arg\,\max_{c_k}](Markdown数学公式语法/max_{c_k}.svg)
 `$arg\,\min_{c_k}$`：![arg,\min_{c_k}](https://math.jianshu.com/math?formula=arg%2C%5Cmin_%7Bc_k%7D)
 `$\mathop {argmin}_{c_k}$`：![\mathop {argmin}_{c_k}](https://math.jianshu.com/math?formula=%5Cmathop%20%7Bargmin%7D_%7Bc_k%7D)
 `$\mathop {argmax}_{c_k}$`：![\mathop {argmax}_{c_k}](https://math.jianshu.com/math?formula=%5Cmathop%20%7Bargmax%7D_%7Bc_k%7D)
 `$\max_{c_k}$`：![\max_{c_k}](https://math.jianshu.com/math?formula=%5Cmax_%7Bc_k%7D)
 `$\min_{c_k}$`：![\min_{c_k}](https://math.jianshu.com/math?formula=%5Cmin_%7Bc_k%7D)

# 6 分式与根式

## 6.1. 分式

第一种，使用`\frac ab`，表示为:![\frac ab](Markdown数学公式语法/frac ab.svg) ，`\frac`作用于其后的两个组a ，b ，结果为。如果你的分子或分母不是单个字符，请使用{…}来分组，比如`$\frac {a+c+1}{b+c+2}$`表示:![\frac {a+c+1}{b+c+2}](Markdown数学公式语法/frac {a%2Bc%2B1}{b%2Bc%2B2}.svg)
 第二种，使用\over来分隔一个组的前后两部分，如`${a+1\over b+1}$`：![{a+1\over b+1}](Markdown数学公式语法/over b%2B1}.svg)

## 6.2. 连分数

书写连分数表达式时，请使用`\cfrac`代替`\frac`或者`\over`两者效果对比如下：
 `\frac` 表示如下：

```markdown
 $$x=a_0 + \frac {1^2}{a_1 + \frac {2^2}{a_2 + \frac {3^2}{a_3 + \frac {4^2}{a_4 + ...}}}}$$

```

显示如下：
 ![x=a_0 + \frac {1^2}{a_1 + \frac {2^2}{a_2 + \frac {3^2}{a_3 + \frac {4^2}{a_4 + ...}}}}](Markdown数学公式语法/frac {4^2}{a_4 %2B ...svg)
 `\cfrac`表示如下：

```markdown
$$x=a_0 + \cfrac {1^2}{a_1 + \cfrac {2^2}{a_2 + \cfrac {3^2}{a_3 + \cfrac {4^2}{a_4 + ...}}}}$$

```

显示如下：
 ![x=a_0 + \cfrac {1^2}{a_1 + \cfrac {2^2}{a_2 + \cfrac {3^2}{a_3 + \cfrac {4^2}{a_4 + ...}}}}](Markdown数学公式语法/cfrac {4^2}{a_4 %2B ...svg)

## 6.3.根式

根式使用`\sqrt` 来表示。
 如开4次方：`$\sqrt[4]{\frac xy}$` 可表示：![\sqrt[4]{\frac xy}](Markdown数学公式语法/frac xy}.svg)
 开平方：`$\sqrt {a+b}$`可表示：![\sqrt {a+b}](Markdown数学公式语法/sqrt {a%2Bb}.svg)

# 7 多行表达式

## 7.1. 分类表达式

定义函数的时候经常需要分情况给出表达式，使用\begin{cases}…\end{cases} 。其中：
 使用`\\` 来分类，
 使用`&`指示需要对齐的位置，
 使用`\ +space`表示空格。
 如：

```markdown
$$
f(n)
\begin{cases}
\cfrac n2, &if\ n\ is\ even\\
3n + 1, &if\  n\ is\ odd
\end{cases}
$$

```

表示:
 ![f(n) \begin{cases} \cfrac n2, &if\ n\ is\ even\\ 3n + 1, &if\ n\ is\ odd \end{cases}](Markdown数学公式语法/end{cases}.svg)
 以及:

```markdown
$$
L(Y,f(X)) =
\begin{cases}
0, & \text{Y = f(X)}  \\
1, & \text{Y $\neq$ f(X)}
\end{cases}
$$

```

```ruby

```

表示:
 ![L(Y,f(X)) = \begin{cases} 0, & \text{Y = f(X)} \\ 1, & \text{Y $\neq$ f(X)} \end{cases}](Markdown数学公式语法/math.svg)%20%3D%20%5Cbegin%7Bcases%7D%200%2C%20%26%20%5Ctext%7BY%20%3D%20f(X)%7D%20%5C%5C%201%2C%20%26%20%5Ctext%7BY%20%24%5Cneq%24%20f(X)%7D%20%5Cend%7Bcases%7D)

如果想分类之间的垂直间隔变大，可以使用`\\[2ex]`代替`\\`来分隔不同的情况。`(3ex,4ex` 也可以用，`1ex`相当于原始距离）。如下所示：

```markdown
$$
L(Y,f(X)) =
\begin{cases}
0, & \text{Y = f(X)} \\[5ex]
1, & \text{Y $\neq$ f(X)}
\end{cases}
$$

```

表示：
 ![L(Y,f(X)) = \begin{cases} 0, & \text{Y = f(X)} \\[5ex] 1, & \text{Y $\neq$ f(X)} \end{cases}](Markdown数学公式语法/math.svg)%20%3D%20%5Cbegin%7Bcases%7D%200%2C%20%26%20%5Ctext%7BY%20%3D%20f(X)%7D%20%5C%5C%5B5ex%5D%201%2C%20%26%20%5Ctext%7BY%20%24%5Cneq%24%20f(X)%7D%20%5Cend%7Bcases%7D)

## 7.2. 多行表达式

有时候需要将一行公式分多行进行显示。

```markdown
$$
\begin{aligned}
\sqrt{37} & = \sqrt{\frac{73^2-1}{12^2}} \\
 & = \sqrt{\frac{73^2}{12^2}\cdot\frac{73^2-1}{73^2}} \\ 
 & = \sqrt{\frac{73^2}{12^2}}\sqrt{\frac{73^2-1}{73^2}} \\
 & = \frac{73}{12}\sqrt{1 - \frac{1}{73^2}} \\ 
 & \approx \frac{73}{12}\left(1 - \frac{1}{2\cdot73^2}\right)
\end{aligned}
$$

```

表示:
 ![\begin{aligned} \sqrt{37} & = \sqrt{\frac{73^2-1}{12^2}} \\ & = \sqrt{\frac{73^2}{12^2}\cdot\frac{73^2-1}{73^2}} \\ & = \sqrt{\frac{73^2}{12^2}}\sqrt{\frac{73^2-1}{73^2}} \\ & = \frac{73}{12}\sqrt{1 - \frac{1}{73^2}} \\ & \approx \frac{73}{12}\left(1 - \frac{1}{2\cdot73^2}\right) \end{aligned}](Markdown数学公式语法/end{aligned}.svg)

```markdown
$$
\begin{aligned}
a&=b+c-d \\
&=e-f \\
&=i \\
\end{aligned}
$$

```

表示:
 ![\begin{aligned} a&=b+c-d \\ &=e-f \\ &=i \\ \end{aligned}](Markdown数学公式语法/end{aligned}-170115307251129.svg)

其中`begin{equation}` 表示开始方程，`end{equation}`表示方程结束；`begin{split}` 表示开始多行公式，`end{split}` 表示结束；公式中用`\\` 表示回车到下一行，`&` 表示对齐的位置。

# 8 方程组

使用\begin{array}...\end{array} 与\left \与\right 配合表示方程组,如:

```markdown
$$
\left \{ 
\begin{array}{c}
a_1x+b_1y+c_1z=d_1 \\ 
a_2x+b_2y+c_2z=d_2 \\ 
a_3x+b_3y+c_3z=d_3
\end{array}
\right.
$$

```

表示：
 ![\left \{ \begin{array}{c} a_1x+b_1y+c_1z=d_1 \\ a_2x+b_2y+c_2z=d_2 \\ a_3x+b_3y+c_3z=d_3 \end{array} \right.](Markdown数学公式语法/right.svg)

注意：通常MathJax通过内部策略自己管理公式内部的空间，因此`a…b` 与`a…….b`（.表示空格）都会显示为`ab`。可以通过在`ab`间加入`\`,增加些许间隙，`\;`增加较宽的间隙，`\quad` 与`\qquad` 会增加更大的间隙。

# 9 特殊函数与符号

## 9.1. 三角函数

`$\sin x$` : ![\sin x](Markdown数学公式语法/sin x.svg)
 `$\arctan x$` : ![\arctan x](Markdown数学公式语法/arctan x.svg)

## 9.2.比较运算符

小于`(\lt )`：![(\lt )](Markdown数学公式语法/lt.svg)
 大于`(\gt )`：![(\gt](https://math.jianshu.com/math?formula=(%5Cgt)
 小于等于`(\le )`：![(\le )](https://math.jianshu.com/math?formula=(%5Cle%20))
 大于等于`(\ge )`：![(\ge )](Markdown数学公式语法/ge.svg)
 不等于`(\ne )` :![(\ne )](Markdown数学公式语法/ne.svg)
 可以在这些运算符前面加上`\not` ，如`\not\lt` : ![\not\lt](Markdown数学公式语法/lt.svg)

## 9.3.集合关系与运算

并集`(\cup)`: ![(\cup)](Markdown数学公式语法/cup.svg)
 交集`(\cap)`: ![(\cap)](Markdown数学公式语法/cap.svg)
 差集`(\setminus)`:![(\setminus)](Markdown数学公式语法/setminus.svg)
 子集`(\subset)`: ![(\subset)](Markdown数学公式语法/subset.svg)
 子集`(\subseteq)`: ![(\subseteq)](Markdown数学公式语法/subseteq.svg)
 非子集`(\subsetneq)`: ![(\subsetneq)](Markdown数学公式语法/subsetneq.svg)
 父集`(\supset)`: ![(\supset)](Markdown数学公式语法/supset.svg)
 属于`(\in)`: ![(\in)](Markdown数学公式语法/in.svg)
 不属于`(\notin)`:![`(\notin)](https://math.jianshu.com/math?formula=%60(%5Cnotin))
 空集`(\emptyset)`: ![(\emptyset)](https://math.jianshu.com/math?formula=(%5Cemptyset))
 空`(\varnothing)`: ![(\varnothing)](Markdown数学公式语法/varnothing.svg)

## 9.4. 排列

`\binom{n+1}{2k}` : ![\binom{n+1}{2k}](Markdown数学公式语法/binom{n%2B1}{2k}.svg)
 `{n+1 \choose 2k}` : ![{n+1 \choose 2k}](Markdown数学公式语法/choose 2k}.svg)

## 9.5. 箭头

`(\to)`:![(\to)](Markdown数学公式语法/to.svg)
 `(\rightarrow)`: ![(\rightarrow)](Markdown数学公式语法/rightarrow.svg)
 `(\leftarrow)`: ![(\leftarrow)](Markdown数学公式语法/leftarrow.svg)
 `(\Rightarrow)`:![`(\Rightarrow)](https://math.jianshu.com/math?formula=%60(%5CRightarrow))
 `(\Leftarrow)`: ![(\Leftarrow)](https://math.jianshu.com/math?formula=(%5CLeftarrow))
 `(\mapsto)`: ![\mapsto)](Markdown数学公式语法/mapsto.svg))

## 9.6. 逻辑运算符

`(\land)`: ![(\land)](Markdown数学公式语法/land.svg)
 `(\lor)`: ![(\lor)](Markdown数学公式语法/lor.svg)
 `(\lnot)`: ![(\lnot)](Markdown数学公式语法/lnot.svg)
 `(\forall)`: ![(\forall)](Markdown数学公式语法/forall.svg)
 `(\exists)`: ![(\exists)](https://math.jianshu.com/math?formula=(%5Cexists))
 `(\top)`: ![(\top)](https://math.jianshu.com/math?formula=(%5Ctop))
 `(\bot)`: ![(\bot)](https://math.jianshu.com/math?formula=(%5Cbot))
 `(\vdash)`: ![(\vdash)](https://math.jianshu.com/math?formula=(%5Cvdash))
 `(\vDash)`:![(\vDash)](https://math.jianshu.com/math?formula=(%5CvDash))

## 9.7.操作符

`(\star)`: ![`(\star)](https://math.jianshu.com/math?formula=%60(%5Cstar))
 `(\ast)`: ![(\ast)](https://math.jianshu.com/math?formula=(%5Cast))
 `(\oplus)`: ![(\oplus)](https://math.jianshu.com/math?formula=(%5Coplus))
 `(\circ)`: ![(\circ)](https://math.jianshu.com/math?formula=(%5Ccirc))
 `(\bullet)`: ![(\bullet)](https://math.jianshu.com/math?formula=(%5Cbullet))

## 9.8.等于

`(\approx)`:![(\approx)](https://math.jianshu.com/math?formula=(%5Capprox))
 `(\sim)`: ![(\sim)](https://math.jianshu.com/math?formula=(%5Csim))
 `(\equiv)`: ![(\equiv)](https://math.jianshu.com/math?formula=(%5Cequiv))
 `(\prec)`: ![(\prec)](https://math.jianshu.com/math?formula=(%5Cprec))

## 9.9.范围

`(\infty)`:![(\infty)](https://math.jianshu.com/math?formula=(%5Cinfty))
 `(\aleph_o)`:![(\aleph_o)](https://math.jianshu.com/math?formula=(%5Caleph_o))
 `(\nabla)`: ![(\nabla)](https://math.jianshu.com/math?formula=(%5Cnabla))
 `(\Im)`: ![(\Im)](https://math.jianshu.com/math?formula=(%5CIm))
 `(\Re)`: ![(\Re)](https://math.jianshu.com/math?formula=(%5CRe))

## 9.10. 模运算

`(\pmod)`: ![`(\pmod)](https://math.jianshu.com/math?formula=%60(%5Cpmod))
 如a \equiv b \pmod n 表示为: ![a \equiv b \pmod n](https://math.jianshu.com/math?formula=a%20%5Cequiv%20b%20%5Cpmod%20n)

## 9.11. 点

`(\ldots)`: ![(\ldots)](https://math.jianshu.com/math?formula=(%5Cldots))
 `(\cdots)`: ![(\cdots)](https://math.jianshu.com/math?formula=(%5Ccdots))
 `(\cdot)`: ![(\cdot)](https://math.jianshu.com/math?formula=(%5Ccdot))
 其区别是点的位置不同，`\ldots` 位置稍低，`\cdots` 位置居中。

```markdown
$$
\begin{cases}
a_1+a_2+\ldots+a_n \\ 
a_1+a_2+\cdots+a_n \\
\end{cases}
$$

```

表示(注意两部分点的位置)：
 ![\begin{cases} a_1+a_2+\ldots+a_n \\ a_1+a_2+\cdots+a_n \\ \end{cases}](Markdown数学公式语法/end{cases}-1701153132684105.svg)

## 9.12.顶部符号

对于单字符，`\hat x`：![\hat x](Markdown数学公式语法/hat x.svg)
 多字符可以使用`\widehat {xy}`：![\widehat {xy}](Markdown数学公式语法/widehat {xy}.svg)
 类似的还有`\overline x`: ![\overline x](Markdown数学公式语法/overline x.svg)
 矢量`\vec x`:![\vec x](Markdown数学公式语法/vec x.svg)
 向量`\overrightarrow {xy}`: ![\overrightarrow {xy}](Markdown数学公式语法/overrightarrow {xy}.svg)
 `\dot x` : ![\dot x](Markdown数学公式语法/dot x.svg)
 `\ddot x`: ![\ddot x](Markdown数学公式语法/ddot x.svg)
 `\dot {\dot x}`: ![\dot {\dot x}](Markdown数学公式语法/dot x}.svg)

# 10 表格

使用`\begin{array}{列样式}…\end{array}`这样的形式来创建表格，列样式可以是`clr` 表示居中，左，右对齐，还可以使用`|`表示一条竖线。表格中各行使用\ 分隔，各列使用& 分隔。使用`\hline` 在本行前加入一条直线。 例如:

```markdown
$$
\begin{array}{c|lcr}
n & \text{Left} & \text{Center} & \text{Right} \\
\hline
1 & 0.24 & 1 & 125 \\
2 & -1 & 189 & -8 \\
3 & -20 & 2000 & 1+10i \\
\end{array}
$$

```

得到：
 ![\begin{array}{c|lcr} n & \text{Left} & \text{Center} & \text{Right} \\ \hline 1 & 0.24 & 1 & 125 \\ 2 & -1 & 189 & -8 \\ 3 & -20 & 2000 & 1+10i \\ \end{array}](Markdown数学公式语法/end{array}.svg)

# 12 汉字、字体与格式

1. 汉字形式，符号：`\mbox{}`，如：![V_{\mbox{初始}}](Markdown数学公式语法/mbox{初始}}.svg)
2. 字体控制，符号：`\displaystyle`，如：![\displaystyle \frac{x+y}{y+z}](Markdown数学公式语法/frac{x%2By}{y%2Bz}.svg)
3. 下划线符号，符号：`\underline`，如：![\underline{x+y}](https://math.jianshu.com/math?formula=%5Cunderline%7Bx%2By%7D)
4. 标签，符号`\tag{数字}`，如：![\tag{11}](https://math.jianshu.com/math?formula=%5Ctag%7B11%7D)
5. 上大括号，符号：`\overbrace{算式}`，如：![\overbrace{a+b+c+d}^{2.0}](Markdown数学公式语法/overbrace{a%2Bb%2Bc%2Bd}^{2.svg)
6. 下大括号，符号：`\underbrace{算式}`，如：![a+\underbrace{b+c}_{1.0}+d](Markdown数学公式语法/underbrace{b%2Bc}_{1.svg)
7. 上位符号，符号：`\stacrel{上位符号}{基位符号}`，如：![\vec{x}\stackrel{\mathrm{def}}{=}{x_1,\dots,x_n}](Markdown数学公式语法/dots%2Cx_n}.svg)

# 13 占位符

1. 两个quad空格，符号：`\qquad`，如：![x \qquad y](https://math.jianshu.com/math?formula=x%20%5Cqquad%20y)
2. quad空格，符号：`\quad`，如：![x \quad y](Markdown数学公式语法/quad y.svg)
3. 大空格，符号`\`，如：![x \ y](Markdown数学公式语法/ y.svg)
4. 中空格，符号`\:`，如：![x : y](Markdown数学公式语法/math-1701153176795124.svg)
5. 小空格，符号`\,`，如：![x , y](Markdown数学公式语法/math-1701153176795125.svg)
6. 没有空格，符号``，如：![xy](Markdown数学公式语法/math-1701153176795126.svg)
7. 紧贴，符号`\!`，如：![x ! y](Markdown数学公式语法/math-1701153176795127.svg)

# 14 定界符与组合

1. 括号，符号：`（）\big(\big) \Big(\Big) \bigg(\bigg) \Bigg(\Bigg)`，如：![（）\big(\big) \Big(\Big) \bigg(\bigg) \Bigg(\Bigg)](https://math.jianshu.com/math?formula=%EF%BC%88%EF%BC%89%5Cbig(%5Cbig)%20%5CBig(%5CBig)%20%5Cbigg(%5Cbigg)%20%5CBigg(%5CBigg))
2. 中括号，符号：`[]`，如：![[x+y]](https://math.jianshu.com/math?formula=%5Bx%2By%5D)
3. 大括号，符号：`\{ \}`，如：![{x+y}](https://math.jianshu.com/math?formula=%7Bx%2By%7D)
4. 自适应括号，符号：`\left \right`，如：`$\left(x\right)$`，![\left(x\right)](https://math.jianshu.com/math?formula=%5Cleft(x%5Cright))
5. 组合公式，符号：`{上位公式 \choose 下位公式}`，如：![{n+1 \choose k}={n \choose k}+{n \choose k-1}](https://math.jianshu.com/math?formula=%7Bn%2B1%20%5Cchoose%20k%7D%3D%7Bn%20%5Cchoose%20k%7D%2B%7Bn%20%5Cchoose%20k-1%7D)
6. 组合公式，符号：`{上位公式 \atop 下位公式}`，如：![\sum_{k_0,k_1,\ldots>0 \atop k_0+k_1+\cdots=n}A_{k_0}A_{k_1}\cdots](https://math.jianshu.com/math?formula=%5Csum_%7Bk_0%2Ck_1%2C%5Cldots%3E0%20%5Catop%20k_0%2Bk_1%2B%5Ccdots%3Dn%7DA_%7Bk_0%7DA_%7Bk_1%7D%5Ccdots)

# 15 四则运算

1. 加法运算，符号：`+`，如：![x+y=z](https://math.jianshu.com/math?formula=x%2By%3Dz)
2. 减法运算，符号：`-`，如：![x-y=z](https://math.jianshu.com/math?formula=x-y%3Dz)
3. 加减运算，符号：`\pm`，如：![x \pm y=z](https://math.jianshu.com/math?formula=x%20%5Cpm%20y%3Dz)
4. 减甲运算，符号：`\mp`，如：![x \mp y=z](https://math.jianshu.com/math?formula=x%20%5Cmp%20y%3Dz)
5. 乘法运算，符号：`\times`，如：![x \times y=z](https://math.jianshu.com/math?formula=x%20%5Ctimes%20y%3Dz)
6. 点乘运算，符号：`\cdot`，如：![x \cdot y=z](https://math.jianshu.com/math?formula=x%20%5Ccdot%20y%3Dz)
7. 星乘运算，符号：`\ast`，如：![x \ast y=z](https://math.jianshu.com/math?formula=x%20%5Cast%20y%3Dz)
8. 除法运算，符号：`\div`，如：![x \div y=z](https://math.jianshu.com/math?formula=x%20%5Cdiv%20y%3Dz)
9. 斜法运算，符号：`/`，如：![x/y=z](https://math.jianshu.com/math?formula=x%2Fy%3Dz)
10. 分式表示，符号：`\frac{分子}{分母}`，如：![\frac{x+y}{y+z}](Markdown数学公式语法/frac{x%2By}{y%2Bz}-1701153176796133.svg)
11. 分式表示，符号：`{分子} \voer {分母}`，如：![{x+y} \over {y+z}](Markdown数学公式语法/over {y%2Bz}.svg)
12. 绝对值表示，符号：`||`，如：![|x+y|](Markdown数学公式语法/math-1701153176796134.svg)

# 16 高级运算

1. 平均数运算，符号：`\overline{算式}`，如：![\overline{xyz}](Markdown数学公式语法/overline{xyz}.svg)
2. 开二次方运算，符号：`\sqrt`，如：![\sqrt x](Markdown数学公式语法/sqrt x.svg)
3. 开方运算，符号：`\sqrt[开方数]{被开方数}`，如：![\sqrt[3]{x+y}](Markdown数学公式语法/sqrt[3]{x%2By}.svg)
4. 对数运算，符号：`\log`，如：![\log(x)](Markdown数学公式语法/log(x.svg)
5. 极限运算，符号：`\lim`，如：![\lim^{x \to \infty}_{y \to 0}{\frac{x}{y}}](Markdown数学公式语法/frac{x}{y}}.svg)
6. 极限运算，符号：`\displaystyle \lim`，如：![\displaystyle \lim^{x \to \infty}_{y \to 0}{\frac{x}{y}}](Markdown数学公式语法/frac{x}{y}}-1701153176796135.svg)
7. 求和运算，符号：`\sum`，如：![\sum^{x \to \infty}_{y \to 0}{\frac{x}{y}}](Markdown数学公式语法/frac{x}{y}}-1701153176797136.svg)
8. 求和运算，符号：`\displaystyle \sum`，如：![\displaystyle \sum^{x \to \infty}_{y \to 0}{\frac{x}{y}}](https://math.jianshu.com/math?formula=%5Cdisplaystyle%20%5Csum%5E%7Bx%20%5Cto%20%5Cinfty%7D_%7By%20%5Cto%200%7D%7B%5Cfrac%7Bx%7D%7By%7D%7D)
9. 积分运算，符号：`\int`，如：![\int^{\infty}_{0}{xdx}](https://math.jianshu.com/math?formula=%5Cint%5E%7B%5Cinfty%7D_%7B0%7D%7Bxdx%7D)
10. 积分运算，符号：`\displaystyle \int`，如：![\displaystyle \int^{\infty}_{0}{xdx}](https://math.jianshu.com/math?formula=%5Cdisplaystyle%20%5Cint%5E%7B%5Cinfty%7D_%7B0%7D%7Bxdx%7D)
11. 微分运算，符号：`\partial`，如：![\frac{\partial x}{\partial y}](https://math.jianshu.com/math?formula=%5Cfrac%7B%5Cpartial%20x%7D%7B%5Cpartial%20y%7D)
12. 矩阵表示，符号：`\begin{matrix} \end{matrix}`，如：![\left[ \begin{matrix} 1 &2 &\cdots &4\5 &6 &\cdots &8\\vdots &\vdots &\ddots &\vdots\13 &14 &\cdots &16\end{matrix} \right]](Markdown数学公式语法/right].svg)

# 17 逻辑运算

1. 等于运算，符号：`=`，如：![x+y=z](https://math.jianshu.com/math?formula=x%2By%3Dz)
2. 大于运算，符号：`>`，如：![x+y>z](Markdown数学公式语法/math-1701153176797139.svg)
3. 小于运算，符号：`<`，如：![x+y<z](Markdown数学公式语法/math-1701153176797140.svg)
4. 大于等于运算，符号：`\geq`，如：![x+y \geq z](https://math.jianshu.com/math?formula=x%2By%20%5Cgeq%20z)
5. 小于等于运算，符号：`\leq`，如：![x+y \leq z](Markdown数学公式语法/leq z.svg)
6. 不等于运算，符号：`\neq`，如：![x+y \neq z](Markdown数学公式语法/neq z.svg)
7. 不大于等于运算，符号：`\ngeq`，如：![x+y \ngeq z](Markdown数学公式语法/ngeq z.svg)
8. 不大于等于运算，符号：`\not\geq`，如：![x+y \not\geq z](Markdown数学公式语法/geq z-1701153176797141.svg)
9. 不小于等于运算，符号：`\nleq`，如：![x+y \nleq z](Markdown数学公式语法/nleq z.svg)
10. 不小于等于运算，符号：`\not\leq`，如：![x+y \not\leq z](Markdown数学公式语法/leq z-1701153176797142.svg)
11. 约等于运算，符号：`\approx`，如：![x+y \approx z](Markdown数学公式语法/approx z.svg)
12. 恒定等于运算，符号：`\equiv`，如：![x+y \equiv z](Markdown数学公式语法/equiv z.svg)

# 18 集合运算

1. 属于运算，符号：`\in`，如：![x \in y](Markdown数学公式语法/in y.svg)
2. 不属于运算，符号：`\notin`，如：![x \notin y](Markdown数学公式语法/notin y.svg)
3. 不属于运算，符号：`\not\in`，如：![x \not\in y](Markdown数学公式语法/in y-1701153176797143.svg)
4. 子集运算，符号：`\subset`，如：![x \subset y](Markdown数学公式语法/subset y.svg)
5. 子集运算，符号：`\supset`，如：![x \supset y](Markdown数学公式语法/supset y.svg)
6. 真子集运算，符号：`\subseteq`，如：![x \subseteq y](Markdown数学公式语法/subseteq y.svg)
7. 非真子集运算，符号：`\subsetneq`，如：![x \subsetneq y](Markdown数学公式语法/subsetneq y.svg)
8. 真子集运算，符号：`\supseteq`，如：![x \supseteq y](Markdown数学公式语法/supseteq y.svg)
9. 非真子集运算，符号：`\supsetneq`，如：![x \supsetneq y](Markdown数学公式语法/supsetneq y.svg)
10. 非子集运算，符号：`\not\subset`，如：![x \not\subset y](Markdown数学公式语法/subset y-1701153176798144.svg)
11. 非子集运算，符号：`\not\supset`，如：![x \not\supset y](Markdown数学公式语法/supset y-1701153176798145.svg)
12. 并集运算，符号：`\cup`，如：![x \cup y](Markdown数学公式语法/cup y.svg)
13. 交集运算，符号：`\cap`，如：![x \cap y](Markdown数学公式语法/cap y.svg)
14. 差集运算，符号：`\setminus`，如：![x \setminus y](Markdown数学公式语法/setminus y.svg)
15. 同或运算，符号：`\bigodot`，如：![x \bigodot y](Markdown数学公式语法/bigodot y.svg)
16. 同与运算，符号：`\bigotimes`，如：![x \bigotimes y](Markdown数学公式语法/bigotimes y.svg)
17. 实数集合，符号：`\mathbb{R}`，如：`\mathbb{R}`
18. 自然数集合，符号：`\mathbb{Z}`，如：`\mathbb{Z}`
19. 空集，符号：`\emptyset`，如：![\emptyset](Markdown数学公式语法/emptyset.svg)

# 19 数学符号

1. 无穷，符号：`\infty`，如：![\infty](Markdown数学公式语法/infty.svg)
2. 虚数，符号：`\imath`，如：![\imath](Markdown数学公式语法/imath.svg)
3. 虚数，符号：`\jmath`，如：![\jmath](Markdown数学公式语法/jmath.svg)
4. 数学符号，符号`\hat{a}`，如：![\hat{a}](Markdown数学公式语法/hat{a}.svg)
5. 数学符号，符号`\check{a}`，如：![\check{a}](Markdown数学公式语法/check{a}.svg)
6. 数学符号，符号`\breve{a}`，如：![\breve{a}](Markdown数学公式语法/breve{a}.svg)
7. 数学符号，符号`\tilde{a}`，如：![\tilde{a}](Markdown数学公式语法/tilde{a}.svg)
8. 数学符号，符号`\bar{a}`，如：![\bar{a}](Markdown数学公式语法/bar{a}.svg)
9. 矢量符号，符号`\vec{a}`，如：![\vec{a}](Markdown数学公式语法/vec{a}.svg)
10. 数学符号，符号`\acute{a}`，如：![\acute{a}](Markdown数学公式语法/acute{a}.svg)
11. 数学符号，符号`\grave{a}`，如：![\grave{a}](Markdown数学公式语法/grave{a}.svg)
12. 数学符号，符号`\mathring{a}`，如：![\mathring{a}](Markdown数学公式语法/mathring{a}.svg)
13. 一阶导数符号，符号`\dot{a}`，如：![\dot{a}](Markdown数学公式语法/dot{a}.svg)
14. 二阶导数符号，符号`\ddot{a}`，如：![\ddot{a}](Markdown数学公式语法/ddot{a}.svg)
15. 上箭头，符号：`\uparrow`，如：![\uparrow](Markdown数学公式语法/uparrow.svg)
16. 上箭头，符号：`\Uparrow`，如：![\Uparrow](Markdown数学公式语法/Uparrow.svg)
17. 下箭头，符号：`\downarrow`，如：![\downarrow](Markdown数学公式语法/downarrow.svg)
18. 下箭头，符号：`\Downarrow`，如：![\Downarrow](Markdown数学公式语法/Downarrow.svg)
19. 左箭头，符号：`\leftarrow`，如：![\leftarrow](Markdown数学公式语法/leftarrow.svg)
20. 左箭头，符号：`\Leftarrow`，如：![\Leftarrow](Markdown数学公式语法/Leftarrow.svg)
21. 右箭头，符号：`\rightarrow`，如：![\rightarrow](Markdown数学公式语法/rightarrow.svg)
22. 右箭头，符号：`\Rightarrow`，如：![\Rightarrow](Markdown数学公式语法/Rightarrow.svg)
23. 底端对齐的省略号，符号：`\ldots`，如：![1,2,\ldots,n](Markdown数学公式语法/ldots%2Cn.svg)
24. 中线对齐的省略号，符号：`\cdots`，如：![x_1^2 + x_2^2 + \cdots + x_n^2](https://math.jianshu.com/math?formula=x_1%5E2%20%2B%20x_2%5E2%20%2B%20%5Ccdots%20%2B%20x_n%5E2)
25. 竖直对齐的省略号，符号：`\vdots`，如：![\vdots](https://math.jianshu.com/math?formula=%5Cvdots)
26. 斜对齐的省略号，符号：`\ddots`，如：![\ddots](https://math.jianshu.com/math?formula=%5Cddots)

# 20 矩阵

使用`\begin{matrix}…\end{matrix}`这样的形式来表示矩阵，在`\begin` 与`\end`之间加入矩阵中的元素即可。矩阵的行之间使用`\\`分隔，列之间使用`&`分隔，例如:

```markdown
$$
\begin{matrix}
1 & x & x^2 \\
1 & y & y^2 \\
1 & z & z^2 \\
\end{matrix}
$$

```

得到：
 ![\begin{matrix} 1 & x & x^2 \\ 1 & y & y^2 \\ 1 & z & z^2 \\ \end{matrix}](Markdown数学公式语法/end{matrix}.svg)

## 20.1. 括号

如果要对矩阵加括号，可以像上文中提到的一样，使用`\left`与`\right` 配合表示括号符号。也可以使用特殊的matrix 。即替换`\begin{matrix}…\end{matrix} 中matrix 为pmatrix ，bmatrix ，Bmatrix ，vmatrix , Vmatrix` 。
 `pmatrix$\begin{pmatrix}1 & 2 \\ 3 & 4\\ \end{pmatrix}$`:pmatrix![\begin{pmatrix}1 & 2 \\ 3 & 4\\ \end{pmatrix}](Markdown数学公式语法/end{pmatrix}.svg)
 `bmatrix$\begin{bmatrix}1 & 2 \\ 3 & 4\\ \end{bmatrix}$` :bmatrix![\begin{bmatrix}1 & 2 \\ 3 & 4\\ \end{bmatrix}](Markdown数学公式语法/end{bmatrix}.svg)
 `Bmatrix$\begin{Bmatrix}1 & 2 \\ 3 & 4\\ \end{Bmatrix}$` :Bmatrix![\begin{Bmatrix}1 & 2 \\ 3 & 4\\ \end{Bmatrix}](Markdown数学公式语法/end{Bmatrix}.svg)
 `vmatrix$\begin{vmatrix}1 & 2 \\ 3 & 4\\ \end{vmatrix}$` :vmatrix![\begin{vmatrix}1 & 2 \\ 3 & 4\\ \end{vmatrix}](Markdown数学公式语法/end{vmatrix}.svg)
 `Vmatrix$\begin{Vmatrix}1 & 2 \\ 3 & 4\\ \end{Vmatrix}$` :Vmatrix![\begin{Vmatrix}1 & 2 \\ 3 & 4\\ \end{Vmatrix}](Markdown数学公式语法/end{Vmatrix}.svg)
 元素省略:
 可以使用\cdots ：⋯，\ddots：⋱ ，\vdots：⋮ 来省略矩阵中的元素，如：

```markdown
$$
\begin{pmatrix}
1&a_1&a_1^2&\cdots&a_1^n\\
1&a_2&a_2^2&\cdots&a_2^n\\
\vdots&\vdots&\vdots&\ddots&\vdots\\
1&a_m&a_m^2&\cdots&a_m^n\\
\end{pmatrix}
$$


```

表示：
 ![\begin{pmatrix} 1&a_1&a_1^2&\cdots&a_1^n\\ 1&a_2&a_2^2&\cdots&a_2^n\\ \vdots&\vdots&\vdots&\ddots&\vdots\\ 1&a_m&a_m^2&\cdots&a_m^n\\ \end{pmatrix}](Markdown数学公式语法/end{pmatrix}-1701153217261330.svg)

## 20.2. 增广矩阵

增广矩阵需要使用前面的表格中使用到的`\begin{array} ... \end{array}`来实现。

```markdown
$$
\left[  \begin{array}  {c c | c} %这里的c表示数组中元素对其方式：c居中、r右对齐、l左对齐，竖线表示2、3列间插入竖线
1 & 2 & 3 \\
\hline %插入横线，如果去掉\hline就是增广矩阵
4 & 5 & 6
\end{array}  \right]
$$


```

显示为：
 ![\left[ \begin{array} {c c | c} 1 & 2 & 3 \\ \hline 4 & 5 & 6 \end{array} \right]](Markdown数学公式语法/right]-1701153236873333.svg)

# 21 公式标记与引用

使用`\tag{yourtag}`来标记公式，如`$$a=x^2-y^3\tag{1}$$`显示为：
 ![a=x^2-y^3\tag{1}](Markdown数学公式语法/tag{1}.svg)

# 22 字体

## 22.1.黑板粗体字

此字体经常用来表示代表实数、整数、有理数、复数的大写字母。
 `$\mathbb ABCDEF$`：![\mathbb ABCDEF](Markdown数学公式语法/mathbb ABCDEF.svg)
 `$\Bbb ABCDEF$`：![\Bbb ABCDEF](Markdown数学公式语法/Bbb ABCDEF.svg)

## 22.3.黑体字

`$\mathbf ABCDEFGHIJKLMNOPQRSTUVWXYZ$`:![\mathbf ABCDEFGHIJKLMNOPQRSTUVWXYZ](Markdown数学公式语法/mathbf ABCDEFGHIJKLMNOPQRSTUVWXYZ.svg)
 `$\mathbf abcdefghijklmnopqrstuvwxyz$`:![\mathbf abcdefghijklmnopqrstuvwxyz](Markdown数学公式语法/mathbf abcdefghijklmnopqrstuvwxyz.svg)

## 22.3.打印机字体

`$\mathtt ABCDEFGHIJKLMNOPQRSTUVWXYZ$`:![\mathtt ABCDEFGHIJKLMNOPQRSTUVWXYZ](Markdown数学公式语法/mathtt ABCDEFGHIJKLMNOPQRSTUVWXYZ.svg)

# 23 希腊字母

| 字母 | 实现            | 字母 | 实现          |
| ---- | --------------- | ---- | ------------- |
| A    | `A`             | α    | `\alhpa`      |
| B    | `B`             | β    | `\beta`       |
| Γ    | `\Gamma`        | γ    | `\gamma`      |
| Δ    | `\Delta`        | δ    | `\delta`      |
| E    | `E`             | ϵ    | `\epsilon`    |
| Z    | `Z`             | ζ    | `\zeta`       |
| H    | `H`             | η    | `\eta`        |
| Θ    | `\Theta`        | θ    | `\theta`      |
| I    | `I`             | ι    | `\iota`       |
| K    | `K`             | κ    | `\kappa`      |
| Λ    | `\Lambda`       | λ    | `\lambda`     |
| M    | `M`             | μ    | `\mu`         |
| N    | `N`             | ν    | `\nu`         |
| Ξ    | `\Xi`           | ξ    | `\xi`         |
| O    | `O`             | ο    | `\omicron`    |
| Π    | `\Pi`           | π    | `\pi`         |
| P    | `P`             | ρ    | `\rho`        |
| Σ    | `\Sigma`        | σ    | `\sigma`      |
| T    | `T`             | τ    | `\tau`        |
| Υ    | `\Upsilon`      | υ    | `\upsilon`    |
| Φ    | `\Phi`          | ϕ    | `\phi`        |
| X    | `X`             | χ    | `\chi`        |
| Ψ    | `\Psi`          | ψ    | `\psi`        |
| Ω    | `\v`            | ω    | `\omega`      |
| ε    | `$\varepsilon$` | ϑ    | `$\vartheta$` |
| ϖ    | `$\varpi$`      | ϱ    | `$\varrho$`   |
| ς    | `$\varsigma$`   | φ    | `$\varphi$`   |

# 附录：

* [Markdown数学公式语法](https://www.jianshu.com/p/383e8149136c)

