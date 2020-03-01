---
title: 'Depth First Search'
date: 2018-06-30 00:00:00
featured_image: '/images/demo/demo-square.jpg'
excerpt: The Blog for Dynamic Programming
---

### 98. Validate BST

#### 思路：验证BST关键就是某一个节点的值比其左子树所有节点都要大，比起右子树所有节点都要小。根据这一点我们就可以通过递归方法来不断判断。

```C++
bool isValidBST(TreeNode* root) {
    return is_BST(root, LLONG_MIN, LLONG_MAX);
}

bool is_BST(TreeNode* root, long long mn, long long mx){
    if(!root) return true;
    if(root->val >= mx || root->val <= mn)
        return false;
    return is_BST(root->left, mn, root->val) && is_BST(root->right, root->val, mx);
}
```

### 105. Construct Binary Tree from Preorder and Inorder Traversal

#### 思路：前序遍历每个子树的根节点都在最前面，而中序遍历的子树根节点在大约中间位置。因此我们通过前序遍历找到根节点，然后再在中序遍历中找到这个根节点的位置，以其为中心找到该节点的左右子树，再分别处理左右子树，然后不断递归的拆解。

```C++
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    int prestart = 0, preend = preorder.size() - 1;
    int instart = 0, inend = inorder.size() - 1;
    
    return construct(preorder, prestart, preend, inorder, instart, inend);
    
}

TreeNode* construct(vector<int>& preorder, int prestart, int preend, vector<int>& inorder, int instart, int inend){
    if(prestart > preend || instart > inend){return NULL;}
    
    TreeNode* root = new TreeNode(preorder[prestart]);

    int index = -1;
    for(int i = 0; i < inorder.size(); i++){
        if(inorder[i] == root->val){
            index = i;
            break;
        }
    }
    
    root->left = construct(preorder, prestart+1, prestart+index-instart, inorder, instart, index-1);
    root->right = construct(preorder, prestart+index-instart+1, preend, inorder, index+1, inend);
    
    return root;
}
```

### 113. Path Sum II

#### 思路：在每个节点时，分别探索左右子树，在探索的过程中不断求和，到达每个子树底部时（即，没有左右子树了）判断与给定值是否相等，如果相等，加入最终结果。注意在每个递归函数的尾部，即，在递归函数返回之前，将加的节点值减去。

```C++
int current_sum = 0;
vector<vector<int>> result;
vector<int> current;

vector<vector<int>> pathSum(TreeNode* root, int sum) {
    
    dfs(root, sum);
    return result;
}   

void dfs(TreeNode* node, int sum){
    if(!node) return;
    current.push_back(node->val);
    current_sum += node->val;
    
    if(node->left == NULL && node->right == NULL){
        if(current_sum == sum)
            result.push_back(current);
    }
    else{
        dfs(node->left, sum);
        dfs(node->right, sum);
    }
    
    current_sum -= current[current.size()-1];
    current.pop_back();
}
```

### 114. Flatten Binary Tree to Linked List

#### 思路：核心思路就是来到某一个节点时，将该节点的右子树先存在一个中间变量中，然后将该节点的左子树替换该节点的右子树，然后找到现在右子树（左子树过来的）最右端、最底端，将中间变量接上。不断地接到最右端，这样不断递归，所有的左子树全部被移到右边。

```C++
void flatten(TreeNode* root) {
    //if root is leaf or root in NULL
    if(root==NULL)return;
    
    flatten(root->left);
    flatten(root->right);
    
    TreeNode* rightFlatTree = root->right;
    root->right = root->left;
    root->left = NULL;
    
    TreeNode* tempNode = root;
    while(tempNode->right!=NULL)
        tempNode=tempNode->right;
    
    tempNode->right = rightFlatTree;   
}
```
### 366. Find Leaves of Binary Tree

#### 思路：这道题目既然要求从所有叶节点开始逐渐删除树，那么我们如何找到每一个树的所有叶节点并在同一时间将所有叶节点拿出来呢，尤其是给定的树的叶节点并不同一深度，即，有的叶节点位于第三层，有的叶节点位于第二层。解决方法就是我们改变计算深度的方法，从下往上计算深度，也就是所有叶节点算作第一层，然后逐渐向上定义第二层，第三层。。。

```C++
unordered_map<int, vector<int>> depth_map;
vector<vector<int>> findLeaves(TreeNode* root) {
    vector<vector<int>> output;
    int max_depth = bottomUp(root);
    for(int i = 1; i <= max_depth; i++){
        output.push_back(depth_map[i]);
    }
    return output;
}

int bottomUp(TreeNode* root){
    if(!root) return 0;
    
    int depth = max(bottomUp(root->left), bottomUp(root->right)) + 1;
    
    depth_map[depth].push_back(root->val);
    
    return depth;
}
```

## 一些典型的DFS问题

### [417. Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/)

#### 思路：建立两个bool矩阵分别记录可以进入pacific 和 atlantic的位置。然后从网格的四个边缘不断向里面进行DFS，用一个变量记录每个经过的格子的值来比较大小判断是否符合条件。

```C++
vector<vector<int>> pacificAtlantic(vector<vector<int>>& matrix) {
    if(matrix.size()==0) return {};
    int m = matrix.size(), n = matrix[0].size();
    vector<vector<int>> output;
    vector<vector<bool>> pacific(m,vector<bool>(n,false)), atlantic(m,vector<bool>(n,false));
    for(int i = 0; i < m; i++){
        dfs(matrix, matrix[i][0], i, 0, pacific);
        dfs(matrix, matrix[i][n-1], i, n-1, atlantic);
    }
    for(int i = 0; i < n; i++){
        dfs(matrix, matrix[0][i], 0, i, pacific);
        dfs(matrix, matrix[m-1][i], m-1, i, atlantic);
    }
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(pacific[i][j] && atlantic[i][j]){
                output.push_back({i,j});
            }
                
        }
    }
    return output;
}

void dfs(vector<vector<int>>& matrix, int prev, int row, int col, 
            vector<vector<bool>>& visited){
    int m = matrix.size(), n = matrix[0].size();
    if(row < 0 || row >= m || col < 0 || col >= n || visited[row][col] || matrix[row][col] < prev)
        return;
    int curr = matrix[row][col];
    visited[row][col] = true;
    dfs(matrix, curr, row+1, col, visited);
    dfs(matrix, curr, row-1, col, visited);
    dfs(matrix, curr, row, col+1, visited);
    dfs(matrix, curr, row, col-1, visited);
}
```

### 694. Number of Distinct Islands

#### 思路：除了正常的DFS寻找独立的小岛外，还要在每个岛上选择一个基准点，计算每个点到基准点的差用来表达每个岛的形状，用来防止translation 重复。

```C++
vector<vector<int>> dicts {{0,1},{1,0},{0,-1},{-1,0}};
int numDistinctIslands(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();
    vector<vector<bool>> visited(m, vector<bool> (n,false));
    unordered_set<string> result;
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            set<string> str;
            if(grid[i][j] == 1 && !visited[i][j]){
                dfs(grid, visited, i, j, i, j, str); 
                string t = "";
                for(auto s : str) t += s + "_";
                result.insert(t);
            }
        }
    }
    return result.size();
}

void dfs(vector<vector<int>>& grid, vector<vector<bool>>& visited, int i, int j, int r, int c, set<string>& str){
    int m = grid.size(), n = grid[0].size();
    visited[i][j] = true;
    for(auto dict : dicts){
        int x = i + dict[0], y = j + dict[1];
        if(x < 0 || x >= m || y < 0 || y >= n || grid[x][y] == 0 || visited[x][y]) continue;
        string s = to_string(x - r) + "_" + to_string(y - c);
        str.insert(s);
        dfs(grid, visited, x, y, r, c, str);
    }
}
```