package algorithm;

import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by MySsir on 16/8/17.
 */

public class Algorithm {
    /**
     * 一个1~n的自然数的乱序数组，其中缺失了1到n之间的两个数，如何快速找出这两个数？
     * */
    public int[] findMissingTwoNum(int[] nums) {
        if (nums == null || nums.length <= 0)
            return null;
        int[] res = new int[2];
        int xy = 0;
        for (int i = 0; i < nums.length; i++) {
            xy = xy ^ nums[i] ^ (i + 1);
        }
        xy = xy ^ (nums.length + 1) ^ (nums.length + 2);
        int count = 0;
        while ((xy & 1) == 0) {
            xy >>= 1;
            count++;
        }
        for (int i = 0; i < nums.length; i++) {
            if (((nums[i] >> count) & 1) == 0)
                res[0] ^= nums[i];
            else
                res[1] ^= nums[i];
        }
        for (int i = 1; i < nums.length + 3; i++) {
            if (((i >> count) & 1) == 0)
                res[0] ^= i;
            else
                res[1] ^= i;
        }
        return  res;
    }

    /**
     * 一个1~n的自然数的乱序数组，其中缺失了1到n之间的某个数，如何快速找出这个数？
     * */
    public int findMissingOneNum(int[] nums) {
        if (nums == null || nums.length <= 0)
            return -1;
        int res = 0;
        for (int i = 0; i < nums.length; i++) {
            System.out.println((i + 1 )+ ", " + nums[i]);
            res = res ^ nums[i] ^ (i + 1);
        }
        return res ^ (nums.length + 1);
    }
    /**
     * 实现栈溢出
     * */
    public void stackOutOfMemory() {
        stackOutOfMemory();
    }

    /**
     * 实现堆溢出
     * */
    public void heapOutOfMemory() {
        List<int[]> list = new ArrayList<>();
        int time = 1000000;
        while (true) {
            list.add(new int[time]);
        }
    }

    /**
     * 数组中有两个数只出现一次，找出这两个数
     * */
    public int[] getOnlyTwo(int[] nums) {
        if (nums == null || nums.length <= 1)
            return null;
        int[] result = new int[2];
        int temp = 0;
        for (int i = 0; i < nums.length; i++)
            temp ^= nums[i];
        int count = 0;
        while ((temp & 1) == 0) {
            count++;
            temp = temp >> 1;
        }
        for (int i = 0; i < nums.length; i++) {
            if (((nums[i] >> count & 1) == 0)) {
                result[0] ^= nums[i];
            } else {
                result[1] ^= nums[i];
            }
        }
        return result;
    }

    /**
     * 找出数组中只出现一次的数
     * */
    public int getOnlyOne(int[] nums) {
        if (nums == null || nums.length <= 0)
            return -1;
        int res = nums[0];
        if (nums.length > 1) {
            for (int i = 1; i < nums.length; i++)
                res ^= nums[i];
        }
        return res;
    }

    /**
     * 归并排序
     * */
    public void mergeSort(int[] nums, int left, int right) {
        int mid = left + (right - left) / 2;
        if (left < right) {
            mergeSort(nums, left, mid);
            mergeSort(nums, mid + 1, right);
            merge(nums, left, mid, right);
            System.out.println(Arrays.toString(nums));
        }
    }
    private static void merge(int[] nums, int left, int mid, int right) {
        int[] temp = new int[right - left + 1];
        int l = left;
        int r = mid + 1;
        int k = 0;
        //把较小的数先移到新数组中
        while (l <= mid && r <= right) {
            if (nums[l] < nums[r])
                temp[k++] = nums[l++];
            else
                temp[k++] = nums[r++];
        }
        //把左边剩余的数移入数组
        while (l <= mid) {
            temp[k++] = nums[l++];
        }
        //把右边剩余的数移入数组
        while (r <= right) {
            temp[k++] = nums[r++];
        }
        //替换原数组中的内容
        for (int i = 0; i < temp.length; i++) {
            nums[left + i] = temp[i];
        }
    }

    /**
     * 给定字符串s1,s2,s3,判断s3是否可以由s1和s2交叉组成得到
     * */
    public boolean isInterleave(String source1, String source2, String target) {
        if (source1.length() + source2.length() != target.length())
            return false;
        char[] charsOfSource_1 = source1.toCharArray();
        char[] charsOfSource_2 = source2.toCharArray();
        char[] charsOfTarget = target.toCharArray();
        //状态转移方程如下：
        //isFlag[i][j] = (isFlag[i-1][j] && charsOfSource_1[i] = charsOfTarget[i+j])  || (isFlag[i][j-1] && charsOfSource_2[j] = charsOfTarget[i+j]);
        boolean[][] isFlag = new boolean[source1.length() + 1][source2.length() + 1];
        isFlag[0][0] = true;
        for (int i = 0; i < charsOfSource_1.length; i++) {
            if (charsOfSource_1[i] == charsOfTarget[i])
                isFlag[i + 1][0] = true;
        }
        for (int i = 0; i < charsOfSource_2.length; i++) {
            if (charsOfSource_2[i]  == charsOfTarget[i])
                isFlag[0][i + 1] = true;
        }
        for (int i = 1; i <= charsOfSource_1.length; i++) {
            char c1 = charsOfSource_1[i - 1];
            for (int j = 1; j <= charsOfSource_2.length; j++) {
                char c2 = charsOfSource_2[j - 1];
                char cT = charsOfTarget[i + j - 1];
                isFlag[i][j] = (isFlag[i][j - 1] && c2 == cT) || (isFlag[i - 1][j] && c1 == cT);
            }
        }
        return isFlag[charsOfSource_1.length][charsOfSource_2.length];
    }

    /**
     * 利用牛顿法求平方根
     * */
    public double sqrtByNewton (int num, double offSet) {
        if (num < 0)
            return 0;
        double result = 10.0;
        int times = 0;
        while (times < 100) {
            if (Math.abs(result * result - num) < offSet) {
                System.out.println(times);
                return Double.parseDouble(new DecimalFormat("#.00").format(result));
            }
            else {
                times++;
                result = (result + num / result) / 2;
            }
            times++;
        }
        return 0;
    }

    /**
     * 利用二分法开根
     * */
    public double sqrtByBinary(int num, double offSet, int sqrtNum) {
        double low = 0, high = num;
        int times = 0;
        double result = low + (high - low) / 2;
        while (times < 100) {
            if (Math.abs(Math.pow(result, sqrtNum) - num) < offSet) {
                System.out.println(times);
                return result;
            }
            if (Math.pow(result, sqrtNum) > num)
                high = result;
            else
                low = result;
            result = low + (high - low) / 2;
            times++;
        }
        return 0;
    }

    /**
     * 用两个栈实现排序
     * */
    public void sortByStack(Stack<Integer> stack) {
        if (stack == null || stack.size() <= 0)
            return;
        Stack<Integer> temp = new Stack<>();
        while (!stack.isEmpty()) {
            if (!temp.isEmpty()) {
                if (stack.peek() <= temp.peek()) {
                    temp.push(stack.pop());
                } else {
                    int num = stack.pop();
                    //int count = 0;
                    while (num > temp.peek()) {
                        stack.push(temp.pop());
                        //count++;
                    }
                    temp.push(num);
                    /*
                    while (count > 0) {
                        temp.push(stack.pop());
                        count--;
                    }
                    */
                }
            } else {
                temp.push(stack.pop());
            }
        }
        while (!temp.isEmpty()) {
            System.out.println(temp.pop());
        }
    }

    /**
     * 找到字符串中第一个不重复的字符
     * */
    public char getFirstNotRepeatedChar(String string) {
        int[] indexs = new int[128];
        //初始化：默认值为：-1
        for (int i = 0; i < indexs.length; i++)
            indexs[i] = -1;
        for (int i = 0; i < string.length(); i++) {
            char ch = string.charAt(i);
            int index = ch - '0';
            if (indexs[index] == -1) {
                indexs[index] = i;
            } else {
                //重复出现设置为：-2
                indexs[index] = -2;
            }
        }
        int minIndex = -1;
        for (int i = 0; i < indexs.length; i++) {
            if (indexs[i] != -1 && indexs[i] != -2) {
                if (minIndex == -1) {
                    minIndex = indexs[i];
                } else {
                    minIndex = minIndex < indexs[i] ? minIndex : indexs[i];
                }
            }
        }
        return string.charAt(minIndex);
    }

    /**
     * 找到带环链表的入口节点
     * */
    public ListNode getFirstNodeInCycle(ListNode head) {
        if (head == null)
            return null;
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                slow = head;
                while (slow != fast) {
                    slow = slow.next;
                    fast = fast.next;
                }
                return slow;
            }
        }
        return null;
    }

    /**
     * 找到两个链表的第一个公共节点
     * */
    public ListNode findFirstCommonNode(ListNode l1, ListNode l2) {
        if (l1 == null || l2 == null)
            return null;
        int length1 = getListNodeLength(l1);
        int length2 = getListNodeLength(l2);
        ListNode longList;
        ListNode shortList;
        int steps = 0;
        if (length1 > length2) {
            longList = l1;
            shortList = l2;
            steps = length1 - length2;
        } else {
            longList = l2;
            shortList = l1;
            steps = length2 - length1;
        }
        for (int i = 0; i < steps; i++) {
            longList = longList.next;
        }
        while (longList != null && shortList != null && longList != shortList) {
            longList = longList.next;
            shortList = shortList.next;
        }
        return longList;
    }
    private int getListNodeLength(ListNode head) {
        int count = 0;
        if (head == null)
            return 0;
        while (head != null) {
            count++;
            head = head.next;
        }
        return count;
    }

    /**
     * 求二叉树第K层的节点数
     * */
    //递归方法
    public int getTreeKthLevelNodeTotalRecursive(TreeNode root, int KthLevel) {
        if (root == null || KthLevel <= 0)
            return 0;
        if (root != null && KthLevel == 1)
            return 1;
        return getTreeKthLevelNodeTotalRecursive(root.left, KthLevel - 1) + getTreeKthLevelNodeTotalRecursive(root.right, KthLevel - 1);
    }

    //非递归写法
    public int getTreeKthLevelNodeTotalNonRecursive(TreeNode root, int KthLevel) {
        if (root == null)
            return 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int curLevelNodesTotal = 0;
        int curLevel = 0;
        while (!queue.isEmpty()) {
            //当前层数
            curLevel++;
            curLevelNodesTotal = queue.size();
            if (curLevel == KthLevel)
                break;
            int cntNode = 0;
            //将下一层节点入队
            while (cntNode < curLevelNodesTotal) {
                cntNode++;
                TreeNode curNode = queue.poll();
                if (curNode.left != null)
                    queue.add(curNode.left);
                if (curNode.right != null)
                    queue.add(curNode.right);
            }
        }
        if (curLevel == KthLevel)
            return curLevelNodesTotal;
        //如果KthLevel大于树的深度
        return 0;
    }

    /**
     * 剑指Offer
     */

    /**
     * 在某射击场有N个靶，每个靶上都有一个分数，存在score数组中。
     * 击中第i个靶的得分为score[left] * score[i] * score[right]，同时原left和right两个靶变为相邻的靶。
     * 其中得分为0的靶是不能射击的，当left不存在或者不能射击时，得分为 score[i] * score[right]，同理right也遵循此规则;
     * 当left和right都不存在或者不能射击时，得分为score[i]。请计算出击中所有能射击的靶，最多能得多少分？
     */
    public int maxScore(int[] score) {
        if(score == null || score.length <= 1)
            return 0;
        int maxTemp = 0, index = 0;
        for (int i = 0; i < score.length; i++) {
            int temp = 0;
            if(score[i] != 0) {
                if (i - 1 >= 0) {
                    if (score[i - 1] != 0) {
                        if (i + 1 < score.length) {
                            if (score[i + 1] != 0) {
                                temp = score[i-1] * score[i] * score[i+1];
                            } else {
                                temp = score[i-1] * score[i];
                            }
                        } else {
                            temp = score[i-1] * score[i];
                        }
                    } else {
                        if (i + 1 < score.length) {
                            if (score[i + 1] != 0) {
                                temp = score[i] * score[i+1];
                            }
                        }
                    }
                } else {
                    if (i + 1 < score.length) {
                        if (score[i + 1] != 0) {
                            temp = score[i] * score[i+1];
                        }
                    }
                }
            }
            if (temp > maxTemp) {
                maxTemp = temp;
                index = i;
            }
        }

        int[] newScore = new int[score.length - 1];
        for (int i = 0, j = 0; i < score.length; i++) {
            if (i != index) {
                newScore[j++] = score[i];
            }
        }

        return maxTemp + maxScore(newScore);
    }

    /**
     * 判断二叉树是不是平衡二叉树
     * */
    public boolean isBalanced(TreeNode root) {
        if (root == null)
            return true;
        int left = treeDepth(root.left);
        int right = treeDepth(root.right);
        if (Math.abs(left - right) > 1)
            return false;
        return isBalanced(root.left) && isBalanced(root.right);
    }

    /**
     * 面试题39：二叉树的深度
     * 题目：输入一颗二叉树的根结点，求该树的深度
     */
     public int treeDepth(TreeNode root) {
         if (root == null)
             return 0;
         int leftDepth = treeDepth(root.left);
         int rightDepth = treeDepth(root.right);
         return leftDepth > rightDepth ? leftDepth + 1 : rightDepth + 1;
     }

    /**
     * 面试题38：数字在排序数组中出现的次数
     * 题目：统计一个数字在排序数组中出现的次数。
     */
    public int getNumOfK(int[] nums, int k) {
        if (nums == null || nums.length <= 0)
            return 0;
        int first = getFirstK(nums, k);
        int last = getLastK(nums, k);
        if (first > -1 && last > -1)
            return last - first + 1;
        else
            return 0;
    }

    /*
        利用二分查找，找到目标值第一次出现的位置
     */
    private int getFirstK(int[] nums, int value) {
        if (nums == null || nums.length <= 0)
            return -1;
        int left = 0, right = nums.length - 1, mid = 0;
        while (left < right) {
            mid = left + (right - left) / 2;
            if (nums[mid] < value)
                left = mid + 1;
            else
                right = mid;
        }
        if (nums[left] != value)
            return -1;
        else
            return left;
    }
    /*
        利用二分查找，找到目标值最后一次出现的位置
     */
    private int getLastK(int[] nums, int value) {
        if (nums == null || nums.length <= 0)
            return -1;
        int left = 0, right = nums.length - 1, mid = 0;
        while (left < right) {
            //注意！！！
            mid = left + (right - left + 1) / 2;
            if (nums[mid] > value)
                right = mid - 1;
            else
                left = mid;
        }
        if (nums[right] != value)
            return -1;
        else
            return left;
    }

    /**
     * 面试题32：从1到n整数中1出现的次数
     * 题目：输入一个整数，求从1到n这n个整数的十进制表示中1出现的次数。
     */
    public int numberOf1InInteger(int num) {
        //假设输入的数字是21345，第一次分成两个区间：1～1345，1346～21345
        String string = String.valueOf(Math.abs(num));
        int first = string.charAt(0) - '0';
        int length = string.length();
        if (length == 1)
            return (first == 0) ? 0 : 1;
        int numFirstDigit = 0;
        if (first > 1)
            numFirstDigit = (int)Math.pow(10, length - 1);
        else if (first == 1)
            numFirstDigit = Integer.parseInt(string.substring(1)) + 1;
        //参考0～9999的所有数字6一共出现了多少次，1346～21345等价于0～19999之中除了最高位1出现了多少次，采用概率的特点
        int numOtherDigits = first * (int)Math.pow(10, length - 1) * (length - 1) / 10;
        //第一次：Integer.parseInt(string.substring(1)) = 1345
        int numRecursive = numberOf1InInteger(Integer.parseInt(string.substring(1)));
        return numFirstDigit + numOtherDigits + numRecursive;
    }

    /**
     * 面试题31：连续子数组的最大和
     * 题目：输入一个数组，数组里有正数也有负数，求连续子数组的最大和。要求时间复杂度为O(n)。
     */
    public int findGreatestSumOfSubArray(int[] nums) {
        if (nums == null || nums.length <= 0)
            return 0;
        int currentSum = 0;
        int maxSum = 0;
        for (int i = 0; i < nums.length; i++) {
            if (currentSum <= 0)
                currentSum = nums[i];
            else
                currentSum += nums[i];
            if (currentSum > maxSum)
                maxSum = currentSum;
        }
        return maxSum;
    }

    /**
     * LeetCode：Rob问题
     * 问题：Rob是一个小偷，街道上有一排房子，每个房屋可以被偷的价值为用数组表示，但是不能偷连续的两家，求出Rob偷的最大价值。
     */
    public int robHouseMaxValue(int[] nums) {
        if (nums == null || nums.length <= 0)
            return 0;
        if (nums.length == 1)
            return nums[0];
        int maxValue_2 = nums[0];
        int maxValue_1 = nums[1];
        int maxValueFinal = Math.max(maxValue_2, maxValue_1);
        //对于每个房屋i，都有抢和不抢两种状态
        //如果抢最大的value就是i-2处的maxValue + nums[i]
        //如果不抢最大值就是i-1处的maxValue
        //此时maxValue就是这两种状态之间的最大值
        for (int i = 2; i < nums.length; i++) {
            maxValueFinal = Math.max(maxValue_2 + nums[i], maxValue_1);
            maxValue_2 = maxValue_1;
            maxValue_1 = maxValueFinal;
        }
        return maxValueFinal;
    }

    /**
     * 面试题29：数组中出现次数超过一半的数字
     * 题目：数组中有一个数字出现了的次数超过数组长度的一半，请找出这个数字。
     * 解析：为了考虑题目的严谨，多做一次判断，检查目标值是不是真的超过数组元素的一半。
     */
    public int moreThanHalfNum(int[] nums) {
        if (nums == null || nums.length <= 0)
            return -1;
        int result = nums[0];
        int count = 1;
        for (int i = 1; i < nums.length; i++) {
            if (count == 0) {
                result = nums[i];
                count = 1;
            } else if (nums[i] == result)
                count++;
            else {
                count--;
            }
        }
        return checkMoreThanHalfNum(nums, result) ? result : -1;
    }
    private boolean checkMoreThanHalfNum(int[] nums, int target) {
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == target)
                count++;
        }
        return (count * 2 > nums.length) ? true : false;
    }

    /**
     * 面试题28：字符串的排列
     * 题目：输入一个字符串，打印出该字符串中字符的所有排列。例如输入abc，则打印出的所有排列有：abc、acb、bca、cab、cba。
     */
    public void printAllPermutation(char[] chars, int start, int end) {
        if (chars == null || chars.length <= 0)
            return;
        //当数组中只有一个字母进行全排列时，按序输出数组即可
        if (start == end) {
            for (int i = 0; i <= end; i++)
                System.out.print(chars[i]);
            System.out.println();
        } else {
            for (int i = start; i <= end; i++) {
                //交换数组第一个元素与后序元素
                char temp = chars[start];
                chars[start] = chars[i];
                chars[i] = temp;
                //后序元素进行全排列
                printAllPermutation(chars, start + 1, end);
                //将交换后的数组还原
                temp = chars[start];
                chars[start] = chars[i];
                chars[i] = temp;
            }
        }
    }

    /**
     * 面试题27：二叉搜索树与双向链表
     * 题目：输入一颗二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点的指针指向。
     */
    public TreeNode convertBSTToDoubleLinkedlist(TreeNode root) {
        //指向当前转化好的双向链表的最后一个元素
        TreeNode lastNode = null;
        convertNode(root, lastNode);
        TreeNode newHead = lastNode;
        while (newHead != null && newHead.left != null)
            newHead = newHead.left;
        return newHead;
    }
    private static void convertNode(TreeNode root, TreeNode lastNode) {
        if (root == null)
            return;
        TreeNode current = root;
        if (current.left != null)
            convertNode(current.left, lastNode);
        current.left = lastNode;
        if (lastNode != null)
            lastNode.right = current;
        lastNode = current;
        if (current.right != null)
            convertNode(current.right, lastNode);
    }

    /**
     * 面试题25：二叉树中和为某一值的路径
     * 题目：输入一颗二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。从树的根结点开始一直往下一直到叶结点所经过的结点形成一条路径。
     */
    public void findPath(TreeNode root, int target) {
        if (root == null)
            return;
        Stack<Integer> path = new Stack<>();
        int currentSum = 0;
        findPathHelp(root, target, path, currentSum);
    }
    private static void findPathHelp(TreeNode node, int target, Stack<Integer> path, int currentSum) {
        currentSum += node.val;
        path.push(node.val);
        //如果是叶子结点，并且路径上结点的和等于target，打印这条路径
        boolean isLeaf = (node.left == null) && (node.right == null);
        if (currentSum == target && isLeaf) {
            for (Integer i : path) {
                System.out.print(i + " ");
            }
            System.out.println();
        }
        //如果不是叶子结点，则遍历它的子结点
        if (node.left != null)
            findPathHelp(node.left, target, path, currentSum);
        if (node.right != null)
            findPathHelp(node.right, target, path, currentSum);
        //在返回到父结点之前，在路径上删除当前结点
        path.pop();
    }

    /**
     * 面试题24：二叉搜索树的后序遍历
     * 题目：输入一个int数组，判断该数组是不是某个二叉搜索树的后序遍历的结果。
     * @param start 子树开始的位置
     * @param end 子树结束的位置
     */
    public boolean verifySquenceOfBST(int[] sequence, int start, int end) {
        if (sequence == null || sequence.length <= 0)
            return false;
        int root = sequence[end];
        int i = start;
        //在二叉搜索树中左子树的结点小于根结点
        for (; i < end; i++) {
            if (sequence[i] > root)
                break;
        }
        //在二叉搜索树中右子树的结点大于根结点
        int j = i;
        for (; j < end; j++) {
            if (sequence[j] < root)
                return false;
        }
        //判断左子树是不是二叉搜索树
        boolean left = true;
        if (i > 0)
            left = verifySquenceOfBST(sequence, start, i - 1);
        //判断左子树是不是二叉搜索树
        boolean right = true;
        if (i < end - 1)
            right = verifySquenceOfBST(sequence, i, end - 1);
        return (left && right);
    }

    /**
     * 面试题22：栈的压入、弹出序列
     * 题目：输入两个整数序列，第一个表示栈的压入顺序，请判断第二个序列是否为该栈的弹出序列。假设压入的所有数字均没有重复。
     */
    public boolean isPopOrder(int[] push, int[] pop) {
        boolean isPossible = false;
        if (push == null || pop == null || push.length <= 0 || pop.length <= 0 || push.length != pop.length)
            return false;
        int length = pop.length;
        int pushNext = 0, popNext = 0;
        Stack<Integer> stack = new Stack<>();
        while (popNext < length) {
            while (stack.isEmpty() || stack.peek() != pop[popNext]) {
                if (pushNext >= length)
                    break;
                stack.push(push[pushNext]);
                pushNext++;
            }
            if (stack.peek() != pop[popNext])
                break;
            stack.pop();
            popNext++;
        }
        if (stack.isEmpty() && popNext == length)
            isPossible = true;
        return isPossible;
    }

    /**
     * 面试题20：顺时针打印矩阵
     * 题目：输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。
     *      例如：如果输入以下矩阵 1  2  3  4
     *                          5  6  7  8
     *                          9  10 11 12
     *                          13 14 15 16
     *      则依次打印出的数字1，2，3，4，8，12，16，15，14，13，9，5，6，7，11，10。
     * 解析：将矩阵想像成若干个圈。假设矩阵是m*n阶，每一圈开始的位置是(startX,startY)，本题中startX = startY，
     *      可以得出最后一圈的终止条件是columns > startX*2 && rows > startY * 2。
     */
    public void printMatrixClockwisely(int[][] matrix) {
        if (matrix == null)
            return;
        int rows = matrix.length;
        int columns = matrix[0].length;
        if (rows <= 0 || columns <= 0)
            return;
        int start = 0;
        //split every circle
        while (rows > start * 2 && columns > start * 2) {
            printMatrixInCircle(matrix, rows, columns, start);
            start++;
        }
    }
    //print num in every circle
    private void printMatrixInCircle(int[][] matrix, int rows, int columns, int start) {
        int endX = columns - 1 - start;
        int endY = rows - 1 - start;
        //From left to right
        for (int i = start; i <= endX; i++)
            System.out.print(matrix[start][i] + " ");
        //From up to down
        if (start < endY) {
            for (int i = start + 1; i <= endY; i++)
                System.out.print(matrix[i][endX] + " ");
        }
        //From right to left
        if (start < endX && start < endY) {
            for (int i = endY - 1; i >= start; i--)
                System.out.print(matrix[endY][i] + " ");
        }
        //From down to up
        if (start < endX && start < endY - 1) {
            for (int i = endY - 1; i >= start + 1; i--)
                System.out.print(matrix[i][start] + " ");
        }
    }

    /**
     * 面试题19：二叉树的镜像
     * 题目：将一颗二叉树镜像反转。
     * 解析：递归方法好理解，但是效率可能会差一点，分别采用递归和非递归完成函数。
     */
    //递归的解法
    public void mirrorRecursively(TreeNode root) {
        if (root == null || (root.left == null && root.right == null))
            return;
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        if (root.left != null)
            mirrorRecursively(root.left);
        if (root.right != null)
            mirrorRecursively(root.right);
    }

    /*
    //非递归解法
    public void mirrorRecursively(TreeNode root) {
        if (root == null)
            return;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            TreeNode left = node.left;
            node.left = node.right;
            node.right = left;
            if (node.left != null)
                queue.offer(node.left);
            if (node.right != null)
                queue.offer(node.right);
        }
    }
    */

    /**
     * 面试题18：树的子结构
     * 题目：输入两颗二叉树A、B，判断B是不是A的子结构。
     * 解析：第一步在树A中查找与根结点的值一样的结点，如果存在，第二步判断A中这个结点的子树和B中这个结点的子树是否一致。
     *      采用递归的方法，注意null指针和递归终止条件。
     */
    public boolean hasSubtree(TreeNode rootA, TreeNode rootB) {
        boolean result = false;
        if (rootA != null && rootB != null) {
            if (rootA.val == rootB.val)
                result = doesTree1HaveTree2(rootA, rootB);
            if (!result)
                result = hasSubtree(rootA.left, rootB);
            if (!result)
                result = hasSubtree(rootA.right, rootB);
        }
        return result;
    }
    private boolean doesTree1HaveTree2(TreeNode nodeA, TreeNode nodeB) {
        //!!!注意，两个if的判断不能调换顺序，仔细思考。
        if (nodeB == null)
            return true;
        if (nodeA == null)
            return false;
        if (nodeA.val != nodeB.val)
            return false;
        return doesTree1HaveTree2(nodeA.left, nodeB.left) && doesTree1HaveTree2(nodeA.right, nodeB.right);
    }

    /**
     * 面试题17：合并两个排序的链表
     * 题目：输入两个递增排序的链表，合并这两个链表并使新链表中的结点仍按递增排序。
     */
    public ListNode mergeListNode(ListNode head_1, ListNode head_2) {
        if (head_1 == null)
            return head_2;
        else if (head_2 ==null)
            return head_1;
        ListNode newHead = null;
        if (head_1.val > head_2.val) {
            newHead = head_2;
            newHead.next = mergeListNode(head_1, head_2.next);
        } else {
            newHead = head_1;
            newHead.next = mergeListNode(head_1.next, head_2);
        }
        return newHead;
    }

    /**
     * 面试题16：反转链表
     * 题目：定义一个函数，输入一个链表的头结点，反转链表并输出反转后结点的头结点。
     */
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode reversedHead = null;
        ListNode current = head;
        ListNode preNode = null;
        while (current != null) {
            ListNode nextNode = current.next;
            if (nextNode == null)
                reversedHead = current;
            current.next = preNode;
            preNode = current;
            current = nextNode;
        }
        return reversedHead;
    }

    /**
     * 面试题15：链表中倒数第K个节点
     * 题目：输入一个链表，输出该链表中倒数第K个结点。结点编号从1开始
     * 解析：考虑代码的鲁棒性：1、head为空；
     *                     2、链表个数小于K；
     */
    public ListNode findKthToTail(ListNode head, int k) {
        if (head == null || k == 0)
            return null;
        ListNode aheadNode = head;
        ListNode behindNode = head;
        for (int i = 0; i < k - 1; i++) {
            if (aheadNode.next != null)
                aheadNode = aheadNode.next;
            else
                return null;
        }
        while (aheadNode.next != null) {
            aheadNode = aheadNode.next;
            behindNode = behindNode.next;
        }
        return behindNode;
    }

    /**
     * 面试题14：调整数组顺序使奇数位于偶数前面
     * 题目：输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分
     */
    public void reOrder(int[] nums) {
        if (nums == null || nums.length <= 1)
            return;
        int start = 0;
        int end = nums.length - 1;
        while (start < end) {
            //采用isEven()函数，可以提高扩展性问题
            while (start < end && !isEven(nums[start]))
                start++;
            while (start < end && isEven(nums[end]))
                end--;
            if (start != end) {
                nums[start] = nums[start] + nums[end];
                nums[end] = nums[start] - nums[end];
                nums[start] = nums[start] - nums[end];
            }
        }
    }
    //考虑拓展性的问题
    private boolean isEven(int num) {
        return (num & 1) == 0 ? true : false;
    }

    /**
     * 面试题12：打印1到最大的n位数
     * 题目：输入数字n，按顺序打印出从1到最大的n位十进制数。比如输入3，则打印出1、2、3一直到最大的3位数即999。
     */
    public void printToMaxOfNDigits(int n) {
        if (n <= 0)
            return;
        int[] number = new int[n];
        while (!increment(number)) {
            printNumber(number);
        }
    }
    private boolean increment(int[] number) {
        boolean isOverflow = false;
        int carry = 0;
        for (int i = number.length - 1; i >= 0; i--) {
            int digit = number[i] + carry;
            if (i == number.length - 1)
                digit++;
            if (digit >= 10) {
                if (i == 0)
                    isOverflow = true;
                else {
                    digit -= 10;
                    carry = 1;
                    number[i] = digit;
                }
            } else {
                number[i] = digit;
                break;
            }
        }
        return isOverflow;
    }
    private void printNumber(int[] number) {
        boolean isBegin = false;
        for (int i = 0; i < number.length; i++) {
            if (!isBegin && number[i] != 0)
                isBegin = true;
            if (isBegin)
                System.out.print(number[i]);
        }
        System.out.println();
    }

    /**
     * 面试题11：数值的整数次方
     * 题目：实现base的整数N次方，不得使用库函数，考虑大数溢出问题
     */
    public double powerWithUnsignedExponent(double base, int exponent) {
        if (exponent == 0)
            return 1;
        if (exponent == 1)
            return base;
        double result  = powerWithUnsignedExponent(base, exponent >> 1);
        result *= result;
        if ((exponent & 1) == 1)
            result *= base;
        return result;
    }

    /**
     * 面试题10：二进制中1的个数
     * 题目：输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示
     */
    public int numberOf_1(int num) {
        int count = 0;
        while (num != 0) {
            num = (num - 1) & num;
            count++;
        }
        return count;
    }

    /**
     * 面试题9：斐波那契数列
     * 题目：写一个函数，输入n，求斐波那契数列的第n项
     * 斐波那契数列定义： f(n) = {       0,         n = 0
     *                        {       1,         n = 1
     *                        {f(n-1) + f(n-2),  n > 1
     * 解析：契波那契数列问题是典型的递归问题，采用递归理解简单但是效率低，将递归转化为非递归效率提升。
     */
    //非递归写法
    public int fibonacci(int n) {
        int first = 0;
        int result = 1;
        if (n <= 0)
            return 0;
        if (n == 1)
            return 1;
        for (int i = 3; i <= n; i++) {
            result = result + first;
            first = result - first;
        }
        return result;
    }
    /*
    //递归写法
    public int fibonacci(int n) {
        if (n <= 0)
            return 0;
        if (n == 1)
            return 1;
        return fibonacci(n-1) + fibonacci(n-2);
    }
    */

    /**
     * 斐波那契数列的变型应用青蛙跳台阶和2*1矩形填充2*n大矩形
     */
    /*
        青蛙跳台阶变态版
        一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
     */
    public int jumpFloorII(int target) {
        if (target <= 0)
            return 0;
        if (target <= 2)
            return target;
        else
            //return (int)Math.pow(2, target - 1);
            return 2 * jumpFloorII(target - 1);
    }

    /*
        一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
     */
    public int jumpFloor(int target) {
        if (target == 1)
            return 1;
        if (target == 2)
            return 2;
        return jumpFloor(target - 1) + jumpFloor(target - 2);
    }

    /**
     * 面试题8：旋转数组的最小数字
     * 题目：把一个数组最开始的若干个数组搬到数组的末尾 ，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。
     * 解析：排好序的数组进行旋转一定程度上也是有序的，可以考虑二分查找的办法。
     */
    public int minNumberInRotatedArray(int[] nums) {
        if (nums == null || nums.length <= 0)
            return -1;
        int start = 0, end = nums.length - 1;
        int mid = start;
        while (nums[start] >= nums[end]) {
            if (end - start == 1) {
                mid = end;
                break;
            }
            mid = (start + end) / 2;
            //如果下标start、mid和end指向的值相等，则只能顺序查找
            if (nums[start] == nums[end] && nums[start] == nums[mid])
                return minInOrder(nums, start, end);
            if (nums[mid] >= nums[start])
                start = mid;
            else
                end = mid;
        }
        return nums[mid];
    }
    private int minInOrder(int[] nums, int start, int end) {
        int result = nums[start];
        for (int i = start + 1; i <= end; i++) {
            if (result > nums[i]) {
                result = nums[i];
                break;
            }
        }
        return result;
    }

    /**
     * 面试题6：重建二叉树
     * 题目：输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树，返回根结点／后序遍历
     * @param preOrder 前序遍历数组
     * @param start 子树起始位置（根结点）
     * @param midOrder 中序遍历数组
     * @param end 中序遍历结束位置
     * @param length 子树结点个数
     * @return 返回根结点root
     */
    public TreeNode rebuildTree(int[] preOrder, int[] midOrder, int start, int end, int length) {
        if (preOrder == null || preOrder.length == 0 || midOrder == null || midOrder.length == 0 || length <= 0)
            return null;
        int value = preOrder[start];
        TreeNode root = new TreeNode(value);
        if (length == 1)
            return root;
        int i = 0;
        while (i < length) {
            if (value == midOrder[end - i])
                break;
            i++;
        }
        root.left = rebuildTree(preOrder, midOrder, start + 1, end - i - 1, length - 1 - i);
        root.right = rebuildTree(preOrder, midOrder, start + length - i, end, i);
        return root;
    }
    /**
     * 二叉树排序树的创建、前序遍历、中序遍历、后序遍历
     */
    public void buildTree(TreeNode root, int val) {
        if (root == null)
            root = new TreeNode(val);
        else {
            if (val < root.val) {
                if (root.left == null)
                    root.left = new TreeNode(val);
                else
                    buildTree(root.left, val);
            } else {
                if (root.right == null)
                    root.right = new TreeNode(val);
                else
                    buildTree(root.right, val);
            }
        }
        return;
    }
    //层序遍历
    public void levelOrder(TreeNode root) {
        if (root == null)
            return;
        Queue<TreeNode> queue = new LinkedList<>();
        TreeNode current = null;
        queue.offer(root);
        while (!queue.isEmpty()) {
            current = queue.poll();
            System.out.print(current.val + " ");
            if (current.left != null)
                queue.offer(current.left);
            if (current.right != null)
                queue.offer(current.right);
        }
    }
    //前(先)序遍历
    public void preOrder(TreeNode root) {
        if (root != null) {
            System.out.print(root.val);
            preOrder(root.left);
            preOrder(root.right);
        }
        return;
    }
    //中序遍历
    public void midOrder(TreeNode root) {
        if (root != null) {
            midOrder(root.left);
            System.out.print(root.val);
            midOrder(root.right);
        }
        return;
    }
    //后序遍历
    public void postOrder(TreeNode root) {
        if (root != null) {
            postOrder(root.left);
            postOrder(root.right);
            System.out.print(root.val);
        }
        return;
    }

    /**
     * 面试题5：从尾到头打印链表
     * 输入一个链表的头结点，从尾到头反过来打印出每个结点的值
     * 解析：可以从头向尾遍历，先压栈再出栈；或者采用递归的思想，递归的方法如果结点规模特别大的情况下，会导致溢出，所以优先考虑使用栈而不是递归的方法。
     */
    public void printListReversingly(ListNode head) {
        if (head != null) {
            if (head.next != null)
                printListReversingly(head.next);
            else
                System.out.println(head.val);
        }
    }

    /*
        面试题4：替换空格
        请实现一个函数，把字符串中的每个空格替换成"%20"。例如输入"We are happy."，则输出"We%20are%20happy."
        解析：从后向前遍历替换，时间复杂度为O(n)
     */
    public char[] replaceBlank(char[] chars) {
        if (chars == null || chars.length <= 0)
            return null;
        int originalLength = chars.length;
        int numOfBlank = 0;
        for (int i = 0; i < originalLength; i++) {
            if (chars[i] == ' ')
                numOfBlank++;
        }
        int newLength = originalLength + numOfBlank * 2;
        char[] result = new char[newLength];
        int index = newLength - 1;
        for (int i = originalLength - 1; i >= 0; i--) {
            if (chars[i] == ' ') {
                result[index--] = '0';
                result[index--] = '2';
                result[index--] = '%';
            } else {
                result[index--] = chars[i];
            }
        }
        return result;
    }

    /*
        面试题3：二维数组中的查找
        在一个二维数组中，每一行都按照从左往右递增的顺序排序，每一列都按照从上往下递增的顺序排序。请完成一个函数，判断二维数组中是否包含给定的目标值。
        解析：优先从右上角或者左下角开始比较，可以每次缩小比较范围
     */
    public int[] findNumInMatrix(int[][] matrix, int target) {
        if (matrix == null)
            return null;
        int[] result = new int[2];
        int rows = matrix.length, columns = matrix[0].length;
        int row = 0, column = columns - 1;
        while (row < rows && column >= 0) {
            if (matrix[row][column] == target) {
                result[0] = row;
                result[1] = column;
                break;
            } else if (matrix[row][column] > target) {
                column--;
            } else {
                row++;
            }
        }
        return result;
    }

    /*
        KMP算法
        解析：首先确定next数组，匹配不成功模式串指针的移动距离：当前匹配成功的个数 - 最后一个匹配成功位置的next数组值
        参考：http://blog.csdn.net/to_be_better/article/details/49086075
     */
    public int kmp(String source, String target) {
        if (target == null || source == null)
            return -1;
        char[] targetArrays = target.toCharArray();
        char[] sourceArrays = source.toCharArray();
        int[] next = getNextArrays(targetArrays);
        for (int i = 0, j = 0; i < sourceArrays.length; i++) {
            while (j > 0 && sourceArrays[i] != targetArrays[j]) {
                j = next[j - 1];
            }
            if (sourceArrays[i] == targetArrays[j])
                j++;
            if (j == targetArrays.length)
                return i - j + 1;
        }
        return -1;
    }
    private int[] getNextArrays(char[] targetArrays) {
        int[] next = new int[targetArrays.length];
        next[0] = 0;
        for (int i = 1, k = 0; i < targetArrays.length; i++) {
            while (k > 0 && targetArrays[k] != targetArrays[i]) {
                //k = next[next[i - 1] - 1];
                k = next[k - 1];
            }
            if (targetArrays[k] == targetArrays[i])
                k++;
            next[i] = k;
        }
        return next;
    }

    /*
        美团面试题
        反转二叉树
     */
    //递归写法
    public TreeNode invertTree(TreeNode root) {
        if (root == null)
            return null;
        invertTreeHelp(root);
        return root;
    }
    public void invertTreeHelp(TreeNode node) {
        if (node == null)
            return;
        TreeNode temp = node.left;
        node.left = node.right;
        node.right = temp;
        invertTreeHelp(node.right);
        invertTreeHelp(node.left);
    }
    /*
    //非递归写法
    public TreeNode invertTree(TreeNode root) {
        if (root == null)
            return null;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            TreeNode left = node.left;
            node.left = node.right;
            node.right = left;
            if (node.left != null)
                queue.offer(node.left);
            if (node.right != null)
                queue.offer(node.right);
        }
        return root;
    }
    */

    /*
        判断两个有环的数组是否存在交集
        解析：存在交集，那么环一定重合，先在其中的一个环上找一个节点，判断这个节点在不在另外一个环上
     */
    public boolean isCycleCross(ListNode l1, ListNode l2) {
        if (l1 == null || l2 == null)
            return false;
        ListNode first = findOneNode(l1);
        ListNode second = findOneNode(l2);
        int times = 0;
        while (times != 2) {
            if (l2 == second)
                times++;
            if (l2 == first)
                return true;
            else
                l2 = l2.next;
        }
        return false;
    }
    //在链表上面找到一个节点
    private ListNode findOneNode(ListNode listNode) {
        ListNode one = listNode;
        ListNode two = listNode;
        while (two.next != null && two.next.next != null) {
            one = one.next;
            two = two.next.next;
            if (one == two)
                return one;
        }
        return null;
    }

    /*
        判断一个链表是不是有环。
        解析：单步走和双步走，如果有环一定会重合
     */
    public boolean isRound(ListNode listNode) {
        if (listNode == null)
            return false;
        ListNode one = listNode;
        ListNode two = listNode;
        while (two.next != null && two.next.next != null) {
            one = one.next;
            two = two.next.next;
            if (one == two)
                return true;
        }
        return false;
    }

    /*
        判断两个链表是否有交集
        解析：如果有交集，两个尾节点一定相等
     */
    public boolean isCross(ListNode l1, ListNode l2) {
        if (l1 == null || l2 == null)
            return false;
        while (l1.next != null) {
            l1 = l1.next;
        }
        while (l2.next != null) {
            l2 = l2.next;
        }
        if (l1 == l2)
            return true;
        else
            return false;
    }

    /*
        Add two numbers
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null)
            return l2;
        if (l2 == null)
            return l1;
        ListNode temp = new ListNode(0);
        ListNode head = temp;
        int carry = 0;
        while (l1 != null || l2 != null || carry != 0) {
            int sum = ((l1 == null) ? 0 : l1.val) + ((l2 == null) ? 0 : l2.val) + carry;
            carry = sum / 10;
            sum = sum % 10;
            ListNode listNode = new ListNode(sum);
            l1 = (l1 == null) ? null : l1.next;
            l2 = (l2 == null) ? null : l2.next;
            temp.next = listNode;
            temp = temp.next;
        }
        //注意返回值，不是返回head，而是head.next，思考一下
        return head.next;
    }

    /*
        堆排序
     */
    public void heapSort(int[] nums) {
        if (nums == null || nums.length <= 1)
            return;
        //build max heap
        for (int i = nums.length/2 - 1; i >= 0; i--)
            maxHeap(nums, i, nums.length);
        for (int i = nums.length - 1; i >= 1; i--) {
            int temp = nums[0];
            nums[0] = nums[i];
            nums[i] = temp;
            maxHeap(nums, 0, i);
        }

    }
    public void maxHeap(int[] nums, int parent, int heapSize) {
        int leftChild = parent * 2 + 1;
        int rightChild = parent * 2 + 2;
        int max = parent;
        if (parent > heapSize / 2)
            return;
        if (leftChild < heapSize && nums[leftChild] > nums[max])
            max = leftChild;
        if (rightChild < heapSize && nums[rightChild] > nums[max])
            max = rightChild;
        if (max != parent) {
            int temp = nums[parent];
            nums[parent] = nums[max];
            nums[max] = temp;
            maxHeap(nums, max, heapSize);
        }
        return;
    }

    /*
        快速排序
     */
    public void quickSort(int[] nums, int left, int right) {
        int i, j, temp;
        i = left;
        j = right;
        if (left > right)
            return;
        temp = nums[left];
        while (i != j) {
            while (nums[j] >= temp && i < j)
                j--;
            if (i < j)
                nums[i++] = nums[j];
            while (nums[i] <= temp && i < j)
                i++;
            if (i < j)
                nums[j--] = nums[i];
        }
        nums[i] = temp;
        quickSort(nums, left, i - 1);
        quickSort(nums, i + 1, right);
    }

    /*
        O(n)的时间复杂度解决two sum问题
     */
    public int[] twoSum(int[] arrays, int target) {
        int[] result = new int[2];
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int i = 0; i < arrays.length; i++) {
            int deviation = target - arrays[i];
            if (hashMap.containsKey(arrays[i]) && hashMap.get(arrays[i]) != i) {
                result[0] = hashMap.get(arrays[i]);
                result[1] = i;
                break;
            }
            hashMap.put(deviation, i);
        }
        return result;
    }

    /*
        输入一个链表，从尾到头打印链表每个节点的值。
     */
    public ArrayList<Integer> printListFromTailToHead(ListNode listnode) {
        Stack<Integer> stack = new Stack();
        while (listnode != null) {
            stack.push(listnode.val);
            listnode = listnode.next;
        }
        ArrayList<Integer> list = new ArrayList<>();
        while (!stack.isEmpty()) {
            list.add(stack.pop());
        }
        return list;
    }

    /*
        请实现一个函数，将一个字符串中的空格替换成“%20”。
        例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
     */
    public String replaceSpace(StringBuffer str) {
        if (str == null || str.length() == 0)
            return "";
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < str.length(); i++) {
            char ch = str.charAt(i);
            if (ch == ' ') {
                result.append("%20");
            } else {
                result.append(ch);
            }
        }
        return result.toString();
    }

    /*
        在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
        请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
     */
    public boolean findInMatrix(int[][] arrays, int target) {
        int len = arrays.length - 1;
        int i = 0;
        while ((len >= 0) && (i < arrays[0].length)) {
            if (arrays[len][i] > target) {
                len--;
            } else if (arrays[len][i] < target) {
                i++;
            }
            else {
                return true;
            }
        }
        return false;
    }

    /**
     * Leetcode
     */

    /**
     * 11.Container with most water
     * 题目：给一个数组，每个非负int元素ai，表示横轴为i，高度为ai，求出两条线与X轴组成水桶最大的体积。
     */
    public int maxArea(int[] heights) {
        if (heights == null || heights.length <= 1)
            return 0;
        int area = 0, left = 0, right = heights.length - 1;
        while (left < right) {
            area = Math.max(area, Math.min(heights[left], heights[right]) * (right - left));
            if (heights[left] > heights[right])
                right--;
            else
                left++;
        }
        return area;
    }

    /*
        39.Combination Sum
     */
    public List<List<Integer>> combinationSum(int[] nums, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        backTracking(result, new ArrayList<>(), 0, target, nums);
        return result;
    }

    public void backTracking(List<List<Integer>> result, List<Integer> current, int from, int target, int[] nums) {
        if (target == 0) {
            List<Integer> list = new ArrayList<>(current);
            result.add(list);
        } else {
            for (int i = from; i < nums.length && nums[i] <= target; i++) {
                current.add(nums[i]);
                backTracking(result, current, i, target - nums[i], nums);
                current.remove(new Integer(nums[i]));
            }
        }
    }

    /*
        35.Search Insert Position
     */
    public int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int index = left + (right - left) / 2;
            if (nums[index] == target)
                return index;
            else if (nums[index] < target)
                left = index + 1;
            else
                right = index - 1;
        }
        return left;
    }

    /*
        34.Search for a Range
     */
    public int[] searchRange(int[] nums, int target) {
        if (nums == null || nums.length <= 0)
            return new int[]{-1, -1};
        int[] result = new int[2];
        //Search first show up
        int left = 0, right = nums.length - 1;
        while (left != right) {
            int head = left + (right - left) / 2;
            if (nums[head] < target)
                left = head + 1;
            else
                right = head;
        }
        if (nums[left] != target) {
            return new int[] {-1, -1};
        }
        else
            result[0] = left;
        //Search last show up
        left = 0;
        right = nums.length - 1;
        while (left != right) {
            //为什么要+1？
            int tail = left + (right - left) / 2 + 1;
            if (nums[tail] > target)
                right = tail - 1;
            else
                left = tail;
        }
        if (nums[left] != target) {
            return new int[] {-1, -1};
        }
        else
            result[1] = left;
        return result;
    }

    /*
        31.Next Permutation
        找出当前数组全排列的下一个子序列
     */
    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length <= 1)
            return;
        int index = nums.length - 2;
        while (index >= 0 && nums[index] >= nums[index + 1])
            index--;
        if (index >= 0) {
            int i = index + 1;
            while (i < nums.length && nums[i] > nums[index])
                i++;
            int temp = nums[i - 1];
            nums[i - 1] = nums[index];
            nums[index] = temp;
        }
        reverse(nums, index + 1);
        return;
    }

    private void reverse(int[] nums, int i) {
        int left = i, right = nums.length - 1;
        while (left < right) {
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            left++;
            right--;
        }
    }

    /*
        29.Divide Two Integers.
        Without using multiplication, division and mod operator.
        解法有问题，边界情况不能正确通过：-2147483648/(-1) 不能返回题目要求的结果
     */
    public int divide(int dividend, int divisor) {
        int sign = ((dividend < 0) ^ (divisor < 0)) ? -1 : 1;
        int dividend_temp = Math.abs(dividend);
        int didisor_temp = Math.abs(divisor);
        int result = 0;
        while (dividend_temp >= didisor_temp) {
            long temp = didisor_temp, multiple = 1;
            while (dividend_temp >= (temp << 1)) {
                temp <<= 1;
                multiple <<= 1;
            }
            dividend_temp -= temp;
            if (result + multiple >= Integer.MAX_VALUE)
                return Integer.MAX_VALUE;
            else
                result += multiple;
        }
        return sign == 1 ? result : -result;
    }

    /*
        22.Generate Parentheses
     */
    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        backtrack(result, "", 0, 0, n);
        return result;
    }

    private void backtrack(List<String> list, String str, int open, int close, int max) {
        if (str.length() == max*2) {
            list.add(str);
            return;
        }
        if (open < max)
            backtrack(list, str+"(", open+1, close, max);
        if (close < open)
            backtrack(list,str+")",open,close+1,max);
    }


    /*
        17.Letter Combinations of a Phone Number
    */
    public List<String> letterCombinations(String digits) {
        LinkedList<String> result = new LinkedList<>();
        if (digits == null || digits.length() == 0)
            return result;
        String[] strs = {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        result.add("");
        for (int i = 0; i < digits.length(); i++) {
            int num = digits.charAt(i) - '0';
            while (result.peek().length() == i) {
                String t = result.remove();
                for (Character ch : strs[num].toCharArray()) {
                    result.add(t + ch);
                }
            }
        }
        return result;
    }

    
    /*
        15.3Sum
        K Sum问题可以退化成 K－1 Sum问题
     */
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        if (nums == null || nums.length <= 2)
            return result;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1])
                continue;
            find(nums, i + 1, nums.length - 1, nums[i], result);
        }
        return result;
    }

    public void find(int[] nums, int left, int right, int target, List<List<Integer>> result) {
        int l = left, r = right;
        while (l < r) {
            if (nums[l] + nums[r] + target == 0) {
                List<Integer> list = new ArrayList<>();
                list.add(target);
                list.add(nums[l]);
                list.add(nums[r]);
                result.add(list);
                while (l < r && nums[l] == nums[l + 1])
                    l++;
                while (l < r && nums[r] == nums[r - 1])
                    r--;
                l++;
                r--;
            }
            else if (nums[l] + nums[r] + target < 0)
                l++;
            else
                r--;
        }
    }

    //12.Integer to Roman
    public String intToRoman(int num) {
        String[] M = {"", "M", "MM", "MMM"};
        String[] C = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
        String[] X = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
        String[] I = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
        return M[num/1000] + C[(num%1000)/100] + X[(num%100)/10] + I[num%10];
    }

    //找到数组中第K大的数,改进快速排序可得出最优解(因为快速排序每次会将一个元素放到最终位置)

    public int findMaxK(int[] nums, int k) {
        if (nums == null || k <= 0 || nums.length < k)
            return -1;
        int length = nums.length;
        int left = 0, right = length - 1;
        int index = partition(nums, left, right);
        boolean flag = true;
        while (flag) {
            if (index == length - k)
                flag = false;
            else if (index < length - k) {
                left = index + 1;
                index = partition(nums, left, right);
            }
            else {
                right = index - 1;
                index = partition(nums, left, right);
            }
        }
        return nums[index];
    }

    private int partition(int[] nums, int left, int right) {
        int l = left, r = right;
        int temp = nums[l];
        while (l != r) {
            while (l < r && nums[r] >= temp)
                r--;
            if (l < r)
                nums[l++] = nums[r];
            while (l < r && nums[l] <= temp)
                l++;
            if (l < r)
                nums[r--] = nums[l];
        }
        nums[l] = temp;
        return l;
    }

    /*
        Get the maximum concurrent pool
        同一时刻，一个桌面池上已经连接的用户数，称为此桌面池的并发连接数。同一个用户，对于同一个桌面池，同一时刻只会有一个连接。
        如果同一个桌面池，一个连接的断开时间恰好是另一个连接的建立时间，不认为这两个连接是并发的。
        输入是N*4的二维数组,表示一共有N个连接, 每个连接由4个Long Integer整数来表示，分别是
        “User ID”, “Pool ID”, “Connection time”, “Disconnection time”. “Connection time”总是小于“Disconnection time“。
        要求是分析这N个连接的数据，找出并发连接数最大的那个桌面池，输出此桌面池的ID。
        如果有多个桌面池并发连接数最大，输出任意其中一个桌面池的id
    */
    //自定义ArrayList<Long>的排序方式
    class MyComparator implements Comparator<Long> {
        public int compare(Long i, Long j) {
            Long abs_i = Math.abs(i);
            Long abs_j = Math.abs(j);
            if (abs_i > abs_j)
                return 1;
            else if (abs_i < abs_j)
                return -1;
            else
                return 0;
        }
    }

    public long getMaxConcurrentPool(long[][] connections) {
        //建立一个map，{key:poolid, value:connection & disconnection}，其中disconnection取反标记
        HashMap<Long, ArrayList<Long>> poolMap = new HashMap<>();
        //遍历二维数组，按照map的结构进行存储
        for (int i = 0; i < connections.length; i++) {
            long poolid = connections[i][1];
            if (poolMap.containsKey(poolid)) {
                ArrayList<Long> item = poolMap.get(poolid);
                item.add(connections[i][2]);
                item.add(-connections[i][3]);
                poolMap.put(poolid, item);
            }
            else {
                ArrayList<Long> item = new ArrayList<>();
                item.add(connections[i][2]);
                item.add(-connections[i][3]);
                poolMap.put(poolid, item);
            }
        }
        //定义最大连接数的poolID，和其对应的最大连接数
        long maxPoolID = 0;
        int max = 0;
        //遍历按照poolid取出相应的所有连接item，对item按照自己的定义的方式排序(按照原有的connection & disconnection time 真实值排序)
        for (Long poolid: poolMap.keySet()) {
            ArrayList<Long> item = poolMap.get(poolid);
            //排序
            Collections.sort(item, new MyComparator());
            int count = thePoolOfMax(item);
            if (count > max) {
                maxPoolID = poolid;
                max = count;
            }
        }
        return maxPoolID;
    }

    //获得当前poolID的最大连接数
    public int thePoolOfMax(ArrayList<Long> list) {
        int count = 0;
        int max = 0;
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) > 0)
                count++;
            else {
                max = max > count ? max : count;
                count = count > 0 ? count - 1 : 0;
            }
        }
        return max;
    }

}