from itertools import product

# all_clf = [pipe1, clf2, pipe3, mv_clf]

# x_min = X_train_std[:, 0].min() - 1
# x_max = X_train_std[:, 0].max() + 1
# y_min = X_train_std[:, 1].min() - 1
# y_max = X_train_std[:, 1].max() + 1

# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     # np.arange(y_min, y_max, 0.1))

# f, axarr = plt.subplots(nrows=2, ncols=2, 
                        # sharex='col', 
                        # sharey='row', 
                        # figsize=(7, 5))

# for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        # all_clf, clf_labels):
    # clf.fit(X_train_std, y_train)
    
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)

    # axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    
    # axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0], 
                                  # X_train_std[y_train==0, 1], 
                                  # c='blue', 
                                  # marker='^',
                                  # s=50)
    
    # axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0], 
                                  # X_train_std[y_train==1, 1], 
                                  # c='red', 
                                  # marker='o',
                                  # s=50)
    
    # axarr[idx[0], idx[1]].set_title(tt)

# plt.text(-3.5, -4.5, 
         # s='Sepal width [standardized]', 
         # ha='center', va='center', fontsize=12)
# plt.text(-10.5, 4.5, 
         # s='Petal length [standardized]', 
         # ha='center', va='center', 
         # fontsize=12, rotation=90)

# plt.tight_layout()
plt.savefig('./figures/voting_panel', bbox_inches='tight', dpi=300)
# plt.show()