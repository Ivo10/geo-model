1. 范围[10, 10, 5]；

2. 输入顶点为258个，前8个顶点为六面体的顶点，后250个为随机生成的顶点；

3. 生成顶点信息：

   ```
   Statistics:
   
     Input points: 50
     Input facets: 12
     Input segments: 12
     Input holes: 0
     Input regions: 0
   
     Mesh points: 2141
     Mesh tetrahedra: 9921
     Mesh faces: 20838
     Mesh faces on exterior boundary: 1992
     Mesh faces on input facets: 1992
     Mesh edges on input segments: 153
     Steiner points on input facets:  849
     Steiner points on input segments:  141
     Steiner points inside domain: 1101
   ```
   
4. 由于z的取值范围为[0,5]，z坐标和地层$f_v$的映射关系定义为：

   | z坐标  | 地层$f_v$ |
   | ------ | --------- |
   | [0, 1) | -0.66     |
   | [1, 2) | -0.33     |
   | [2, 3) | 0         |
   | [3, 4) | 0.33      |
   | [4, 5] | 0.66      |

5. 定义图卷积网络信息

   | 层数      | 信息    |
   | --------- | ------- |
   | 图卷积层1 | 3->16   |
   | 图卷积层2 | 16->64  |
   | 图卷积层3 | 64->128 |
   | 全连接层  | 128->1  |

   ![image-20240731145616389](C:\Users\Xiong Wei\AppData\Roaming\Typora\typora-user-images\image-20240731145616389.png)