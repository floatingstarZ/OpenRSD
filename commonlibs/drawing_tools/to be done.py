# draw iou with different score
fig_name = '/IOU_DISTRIBUTION_Area_%s.png' % a
fig = plt.figure(num=n_interval, figsize=(15, 8), dpi=80)
fig.suptitle('IOU DISTRIBUTION')
W = 5
H = 4
for i in range(H):
    for j in range(W):
        count = i * W + j
        print(count, i, j)
        ax = fig.add_subplot(H, W, count + 1)
        ax.set_title('SCORE %.2f' % (count / n_interval))
        ax.plot(plot_x, D[:, count])
plt.savefig(fig_name)

# draw score with different iou
fig_name = '/SCORE_DISTRIBUTION_Area_%s.png' % a
fig = plt.figure(num=n_interval, figsize=(15, 8), dpi=80)
fig.suptitle('SCORE DISTRIBUTION')
W = 5
H = 4
for i in range(H):
    for j in range(W):
        count = i * H + j
        ax = fig.add_subplot(H, W, count + 1)
        ax.set_title('IOU %.2f' % (count / n_interval))
        ax.plot(plot_x, D[:, count])
plt.savefig(fig_name)

# # 使用add_subplot在窗口加子图，其本质就是添加坐标系
# # 三个参数分别为：行数，列数，本子图是所有子图中的第几个，最后一个参数设置错了子图可能发生重叠
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)
# # 绘制曲线
# ax1.plot(np.arange(0, 1, 0.1), range(0, 10, 1), color='g')
# # 同理，在同一个坐标系ax1上绘图，可以在ax1坐标系上画两条曲线，实现跟上一段代码一样的效果
# ax1.plot(np.arange(0, 1, 0.1), range(0, 20, 2), color='b')
# # 在第二个子图上画图
# ax2.plot(np.arange(0, 1, 0.1), range(0, 20, 2), color='r')