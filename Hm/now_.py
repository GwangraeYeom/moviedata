import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime

def timetodate():
    file = pd.read_csv('../data/ratings.csv')
    frame = pd.DataFrame(file)

    print( datetime.datetime.fromtimestamp(964982400 / 1000))
    print( datetime.datetime.fromtimestamp(964982224 / 1000))
    print(datetime.datetime.fromtimestamp(964984100 / 1000))

    # print(file)
    # print(frame)
    date_data = []
    time_data = []
    hour_data = []
    obj = frame.loc[0:, "timestamp"]
    for i in obj:
        #print(i)
        timetodate = datetime.datetime.fromtimestamp(i / 1000)
        # 연 월 일
        date_data.append(timetodate.strftime("%Y%m%d"))
        # 시간 분
        time_data.append(timetodate.strftime("%H%M"))
        #  시간
        hour_data.append(timetodate.strftime("%H"))
        #print(time_data)
        # print(i)
        # print(i['timestamp'])
    date_series = pd.Series(date_data)
    time_series = pd.Series(time_data)
    hour_series = pd.Series(hour_data)
    # print(date_series)
    # print(time_series)
    frame['date_series'] = date_series
    frame['time_series'] = time_series
    frame['hour_series'] = hour_series
    # print(frame)
    frame.to_csv("../saved_data/movie_genres.csv", index=False)
    #print(obj)
    # date = datetime.datetime.fromtimestamp(964982703/1000)

    # print(date)
    # print(date.ctime())
    # print(date.isoformat())
    print("끝")


def datashow(movieId):
    # 영화별, 장르별
    file = pd.read_csv('../saved_data/movie_genres.csv')
    frame = pd.DataFrame(file)
    # obj1 = frame.loc[:, ["userId", "rating", "movieId", "time_series", "timestamp", "hour_series"]]

    frame = frame[(frame['movieId'] == 296)]

    frame.to_csv("../saved_data/obj1.csv", index=False)

    hour_avg = frame.groupby(frame['hour_series'])[['rating']].mean()
    hour_count = frame.groupby(frame['hour_series'])[['rating']].count()

    user_avg = frame.groupby(['userId', 'hour_series'])[['rating']].mean()

    # temp = obj1[(obj1['time_series'] < a_time) & (obj1['time_series'] >= b_time)]
    # temp = temp.groupby(temp['time_series'])[['rating']].mean()
    # temp.to_csv("../saved_data/temp.csv")
    # temp = pd.read_csv("../saved_data/temp.csv")

    hour_avg.to_csv("../saved_data/hour_temp.csv")
    hour_avg = pd.read_csv("../saved_data/hour_temp.csv")

    hour_count.to_csv("../saved_data/hour_count.csv")
    hour_count = pd.read_csv("../saved_data/hour_count.csv")



    user_avg.to_csv("../saved_data/user_avg.csv")
    user_avg = pd.read_csv("../saved_data/user_avg.csv")
    user_avg = user_avg[(user_avg['userId'] == 1)]

    print(user_avg)
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3, 1, 3)
    #ax1.plot(temp['time_series'],temp['rating'], 'g--')
    #ax1.set_xticks(range(60))
    #ax2.scatter(temp['time_series'],temp['rating'])

    ax1.plot(hour_avg['hour_series'], hour_avg['rating'], 'g--')
    ax1.set_xticks(range(24))
    ax2.plot(hour_count['hour_series'], hour_count['rating'], 'g--')
    #ax2.scatter(hour_avg['hour_series'], hour_avg['rating'])
    ax3.scatter(user_avg['hour_series'], user_avg['rating'])
    ax2.set_xticks(range(24))
    ax3.set_xticks(range(24))
    plt.show()

def naver_show(movieId):
    # 영화별, 장르별
    file = pd.read_csv('./titanic_movie_rating.csv')
    # file = pd.read_csv('./sound_movie_rating.csv')
    frame = pd.DataFrame(file)
    # obj1 = frame.loc[:, ["userId", "rating", "movieId", "time_series", "timestamp", "hour_series"]]

    # frame = frame[(frame['movieId'] == 296)]

    # frame.to_csv("../saved_data/obj1.csv", index=False)

    hour_avg = frame.groupby(frame['hour'])[['review']].mean()
    hour_count = frame.groupby(frame['hour'])[['review']].count()

    # user_avg = frame.groupby(['userId', 'hour_series'])[['rating']].mean()

    # temp = obj1[(obj1['time_series'] < a_time) & (obj1['time_series'] >= b_time)]
    # temp = temp.groupby(temp['time_series'])[['rating']].mean()
    # temp.to_csv("../saved_data/temp.csv")
    # temp = pd.read_csv("../saved_data/temp.csv")

    hour_avg.to_csv("../saved_data/hour_temp.csv")
    hour_avg = pd.read_csv("../saved_data/hour_temp.csv")

    hour_count.to_csv("../saved_data/hour_count.csv")
    hour_count = pd.read_csv("../saved_data/hour_count.csv")



    # user_avg.to_csv("../saved_data/user_avg.csv")
    # user_avg = pd.read_csv("../saved_data/user_avg.csv")
    # user_avg = user_avg[(user_avg['userId'] == 1)]

    # print(user_avg)
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    # ax3 = fig.add_subplot(3, 1, 3)
    #ax1.plot(temp['time_series'],temp['rating'], 'g--')
    #ax1.set_xticks(range(60))
    #ax2.scatter(temp['time_series'],temp['rating'])

    ax1.plot(hour_avg['hour'], hour_avg['review'], 'g--')
    ax1.set_xticks(range(24))
    ax2.plot(hour_count['hour'], hour_count['review'], 'g--')
    #ax2.scatter(hour_avg['hour_series'], hour_avg['rating'])
    # ax3.scatter(user_avg['hour_series'], user_avg['rating'])
    ax2.set_xticks(range(24))
    # ax3.set_xticks(range(24))
    plt.show()


def movie_genre():
    # 영화별, 장르별
    file = pd.read_csv('../data/movies.csv')
    df = pd.DataFrame(file)
    split = df.genres.str.split('|')
    # print(split)
    split = split.apply(lambda x: pd.Series(x))
    # print(split)
    # print(split.stack())
    file = pd.read_csv('../data/movies.csv')
    df = pd.DataFrame(file)
    all_genres = []

    for i in df.genres:
        all_genres.extend(i.split('|'))
    genres = pd.unique(all_genres)
    # print(genres)

    # print(df[['genres', 'movieId']])

    # print(df)
    # print(type(df))
    # df의 header값만 출력된다...     for i in df:
    # 2개 이상의 컬럼은 list로 전달해야
    # for i in df.head().values:
        # print()

    movieid_list = []
    genre_list = []
    title_list = []

    def func(data):
        for i in data.genres.split('|'):
            # print(data.movieId)
            movieid_list.append(data.movieId)
            genre_list.append(i)
            title_list.append(data.title)

    # axis = 1 -> 행 단위, axis = 0 -> 열 단위
    # df[['genres', 'movieId', 'title']].apply(func, axis = 1)
    df[['genres', 'movieId', 'title']].apply(func, axis = 1)

    # print(movieid_list)
    # print(genre_list)
    # print(title_list)

    movieid_series = pd.Series(movieid_list)
    genre_series = pd.Series(genre_list)
    title_series = pd.Series(title_list)
    # print(date_series)
    # print(time_series)
    frame = pd.DataFrame()
    frame['movieid'] = movieid_series
    frame['genres'] = genre_series
    frame['title'] = title_series
    # print(frame)
    frame.to_csv("../data/movie_genres.csv", index=False)

    # 장르별 데이터
    # print(frame[frame['genres'] == 'Adventure'])
    adv_frame = frame[frame['genres'] == 'Adventure'].head(100)
    # print(adv_frame)


    # 평점 데이터
    file = pd.read_csv('../data/ratings.csv')
    rating_data = pd.DataFrame(file).head(1000)
    print(rating_data)
    print("========================")
    # 특정 장르에 대한 평점
    print("oooooooooo   test")
    print(rating_data.shape)
    print(frame.shape)

    print(adv_frame['movieid'].head())

    lst = adv_frame['movieid'].unique()

    def func(data):
        if data in lst:
            return True
        return False

    result = rating_data[rating_data['movieId'].apply(func)]

    print(result.head(15))


    #rating_data = rating_data[rating_data['movieId'] == adv_frame['movieid']]

    print("oooooooooo   test")



    # print(rating_data)

    rating_data = rating_data.groupby(rating_data['movieId'])[['rating']].mean()

    # print(rating_data)
    data = pd.merge(frame, rating_data, left_on='movieid', right_on='movieId')
    # print(data)

def movie_book():
    pd.options.display.max_rows = 10
    unames = ['user_Id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_csv('../data/users.csv', sep="::", header=None, names=unames)

    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('../data/ratings.csv', header=None, names=rnames)

    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_csv('../data/movies.csv', header=None, names=mnames)

    data = pd.merge(pd.merge(ratings, users), movies)

    mean_ratings = data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')
    ratings_by_title = data.groupby('title').size()
    active_titles = ratings_by_title.index[ratings_by_title >= 250]

    mean_ratings = mean_ratings.loc[active_titles]

    top_female_ratings = mean_ratings.sort_values(by='F', ascending=False)
    mean_ratings['diff'] = mean_ratings.sort_values(by='diff')

    # 영화별 표준편차
    rating_std_by_title = data.groupby('title')['rating'].std()

    rating_std_by_title.loc[active_titles]
    rating_std_by_title.sort_values(ascending=False)[:10]


def movie_suggest():

    pd.options.display.max_rows = 100

    file = pd.read_csv('../data/ratings.csv')
    df = pd.DataFrame(file)
    print(df.columns)
    print(df.shape)


    # 영화별 통계
    CountPerMovie = df.groupby(['movieId'],as_index=False)['rating'].count()
    print('---------CountPerMovie--------')
    print(CountPerMovie)
    CountPerMovie = CountPerMovie.sort_values(by='rating', ascending=False)
    print('--------CountPerMovie_Describe--------')
    print(CountPerMovie.describe())

    #유저별 통계
    CountPerUser = df.groupby(['userId'],as_index=False)['rating'].count()
    CountPerUser = CountPerUser.sort_values(by='rating', ascending=False)
    print('---------CountPerUser--------')
    print(CountPerUser)
    print('--------CountPerUser_Describe--------')
    print(CountPerUser.describe())

    #레이팅 10000개 이상과 레이팅 48개 이상 갯수 차이
    million_rated_movies = CountPerMovie[CountPerMovie['rating'] >= 13500]
    print(million_rated_movies)
    million_rated_movies['rating'].plot(kind='bar')


    fe_rated_movies = CountPerMovie[CountPerMovie['rating'] >= 48]
    print(fe_rated_movies)
    fe_rated_movies['rating'].plot(kind='bar')

    plt.show()
    # print(CountPerMovie['rating'].count())
    # CountPerMovie.plot.hist()
    # plt.show()
    #
    # print("+++++++++++++0++++++++++++++")
    # most_rated_movies = movie_count[movie_count['rating'] >= 10000]
    # print(most_rated_movies)
    # print("+++++++++++++1++++++++++++++")
    # data = df[df['movieId'].isin(most_rated_movies.index)]
    # print("+++++++++++++2++++++++++++++")
    # user_ratings = data.pivot_table(index=['userId'], columns=['movieId'], values=['rating'])
    # print("+++++++++++++3++++++++++++++")
    # user_ratings = user_ratings.dropna(thresh=10, axis=1).fillna(0)
    # print("+++++++++++++4++++++++++++++")
    # item_similarity_df = user_ratings.corr(method='pearson')
    # print(item_similarity_df.head(5))
#     # print(item_similarity_df)
#     #item_similarity_df.reset_index(level=1)
#
#     # print(item_similarity_df)
#
#     # print(item_similarity_df.reset_index()['movieId'])
#     new_idx = item_similarity_df.reset_index()['movieId']
#
#     print(item_similarity_df.set_index(new_idx).T.set_index(new_idx).T)
#
#     item_similarity_df = item_similarity_df.set_index(new_idx).T.set_index(new_idx).T
#
#
#     print("++++")
#     item_similarity_df.to_csv('./item_similarity_df.csv',index=True)
#
#
#     item_similarity_df = pd.DataFrame()
#     item_similarity_df = pd.read_csv('./item_similarity_df.csv')
#
#     # print(item_similarity_df)
#
#     new_idx = item_similarity_df.reset_index()['movieId']
#
#     item_similarity_df = item_similarity_df.set_index(new_idx)
#     item_similarity_df = item_similarity_df.drop(columns='movieId')
#     # print(item_similarity_df)
#
#
#     index_list = item_similarity_df.index
#     movie_list = pd.read_csv('../data/movies.csv')
#     # print(movie_list)
#     print("+++++")
#     index_series = movie_list['movieId']
#     movie_list = movie_list.set_index(index_series)
#     print(index_list)
#     temp = pd.DataFrame()
#     temp1 = pd.DataFrame()
#
#     def func(data):
#         if data in index_list:
#             return True
#         return False
#
#     result = movie_list[movie_list['movieId'].apply(func)]
#
#
#     print(result)
#
#     # for i in index_list:
# #        temp = movie_list.pop(i)
#        # print(movie_list.loc[movie_list['movieId'] == i])
#         # temp1 = pd.concat([temp, movie_list.loc[movie_list['movieId'] == i]])
#         # temp1.append(movie_list.loc[movie_list['movieId'] == i])
#         # print(movie_list[movie_list['movieId'] == i])
#         # temp.append(movie_list[movie_list['movieId'] == i])
#
# #         movie_list.drop(i, inplace=True)
#     # print(temp1)
#
#     #print(movie_list)
#     result.to_csv('./movie_list.csv', index=False)
#     # movie_list.to_csv('./movie_list.csv', index=False)
#
#
#     # print(item_similarity_df[('rating', 32)])
#     print("+++++++++++++5++++++++++++++")
#     def get_similar_movies(movie_id, user_rating):
#         print("+++++++++++++9++++++++++++++")
#         # similar_score = item_similarity_df[ item_similarity_df.loc[movieId] == movie_id ] * (user_rating - 2.5)
#         similar_score = item_similarity_df.loc[movie_id] * (user_rating - 2.5)
#         similar_score = similar_score.sort_values(ascending=False)
#
#         return similar_score
#
#     # print(get_similar_movies(32, 5))
#
#
#     # def func(data):
#         # print(data.movieId)
#
#     # 1 -> 행단위로 반복문 0 -> 컬럼 단위로 반복문
#     # movie_list.apply(func, axis=1)
#
#     #for i in list(item_similarity_df.index):
#         #print(movie_list.index)
#
#     print("+++++++++++++6++++++++++++++")
# #    test_data = [(1, 5), (2, 1), (3, 5), (4, 2), (5, 2)]
#     test_data = [(1, 5), (2, 2), (3, 2)]
#     similar_movies = pd.DataFrame()
#     print("+++++++++++++7++++++++++++++")
#     # print(get_similar_movies(32,2))
#     for movie, rating in test_data:
#         similar_movies = similar_movies.append(get_similar_movies(movie, rating))
#     print("+++++++++++++8++++++++++++++")
#     print(similar_movies)
#     print(similar_movies.sum().sort_values(ascending=False).head(6))


def temp1():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import sys
    import pickle
    from surprise import Dataset, Reader, SVD, accuracy
    from surprise.model_selection import train_test_split

    movies = pd.read_csv('../data/movies.csv')
    # genome_scores = pd.read_csv('../data/genome-scores.csv')
    # tags = pd.read_csv('../data/tags.csv')
    # genome_tags = pd.read_csv('../data/genome-tags.csv')
    # Use ratings data to downsample tags data to only movies with ratings
    ratings = pd.read_csv('../data/ratings.csv')
    # print(ratings)
    print("+++++++++++==")
    ratings = ratings.drop_duplicates('movieId')

    # print(ratings)
    # 사용자 추가
    temp_df = pd.DataFrame
    rating_ser = [4.0, 5.0, 2.0]
    movie_ser = [1,2,3]
    user_id = ['138494', '138494', '138494']
    # userId,movieId,rating,timestamp
    # temp_df['userId'] = pd.Series(user_id)
    # temp_df['movieId'] = pd.Series(movie_ser)
    # temp_df['rating'] = pd.Series(rating_ser)

    # pd.concat([ratings, temp_df])


    # instantiate a reader and read in our rating data
    reader = Reader(rating_scale=(1, 5))
    ratings_f = ratings.groupby('userId').filter(lambda x:len(x) >= 55)
    movie_list_rating = ratings_f.movieId.unique().tolist()
    Mapping_file = dict(zip(movies.title.tolist(), movies.movieId.tolist()))
    data = Dataset.load_from_df(ratings_f[['userId', 'movieId', 'rating']], reader)

    # train SVD on 75% of known rates
    trainset, testset = train_test_split(data, test_size=.25)
    algorithm = SVD()
    algorithm.fit(trainset)
    predictions = algorithm.test(testset)

    # check the accuracy using Root Mean Square Error
    accuracy.rmse(predictions)
    print("+++++++++++++1+++++++++++++")
    def pred_user_rating(ui):
        if ui in ratings_f.userId.unique():
            ui_list = ratings_f[ratings_f.userId == ui].movieId.tolist()
            d = {k: v for k, v in Mapping_file.items() if not v in ui_list}
            predictedL = []
            for i, j in d.items():
                predicted = algorithm.predict(ui, j)
                predictedL.append((i, predicted[3]))
            pdf = pd.DataFrame(predictedL, columns=['movies', 'ratings'])
            pdf.sort_values('ratings', ascending=False, inplace=True)
            pdf.set_index('movies', inplace=True)
            return pdf.head(10)
        else:
            print("User Id does not exist in the list!")
            return None

    user_id = 1
    print(pred_user_rating(user_id))


def temp2():
    import sys, gc

    # Load movies data
    movies = pd.read_csv('../data/movies.csv')
    genome_scores = pd.read_csv('../data/genome-scores.csv')
    tags = pd.read_csv('../data/tags.csv')
    genome_tags = pd.read_csv('../data/genome-tags.csv')
    # Use ratings data to downsample tags data to only movies with ratings
    ratings = pd.read_csv('../data/ratings.csv')
    # ratings = ratings.drop_duplicates('movieId')

    ratings_f = ratings.groupby('userId').filter(lambda x: len(x) >= 1000)
    movie_list_rating = ratings_f.movieId.unique().tolist()
    movies = movies[movies.movieId.isin(movie_list_rating)]
    # map movie to id:
    Mapping_file = dict(zip(movies.title.tolist(), movies.movieId.tolist()))
    tags.drop(['timestamp'], 1, inplace=True)
    ratings_f.drop(['timestamp'], 1, inplace=True)

    mixed = pd.DataFrame()
    Final = pd.DataFrame()
    mixed = pd.merge(movies, tags, on='movieId', how='left')

    mixed.fillna("", inplace=True)
    mixed = pd.DataFrame(mixed.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x)))
    Final = pd.merge(movies, mixed, on='movieId', how='left')
    Final['metadata'] = Final[['tag', 'genres']].apply(lambda x: ' '.join(x), axis=1)
    Final[['movieId', 'title', 'metadata']].head(3)

    from sklearn.feature_extraction.text import TfidfVectorizer
    # Creating a content latent matrix from movie metadata:
    # tf-idf vectors and truncated SVD:

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(Final['metadata'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=Final.index.tolist())
    print(tfidf_df.shape)  # (26694, 23704)





    # The first 200 components explain over 50% of the variance:
    # Compress with SVD
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=200)
    latent_matrix = svd.fit_transform(tfidf_df)
    #explained = svd.explained_variance_ratio_.cumsum()
    #plt.plot(explained, '.-', ms=16, color='red')
    #plt.xlabel('Singular value components', fontsize=12)
    #plt.ylabel('Cumulative percent of variance', fontsize=12)
    #plt.show()

    # number of latent dimensions to keep
    n = 200
    latent_matrix_1_df = pd.DataFrame(latent_matrix[:, 0:n], index=Final.title.tolist())

    # our content latent matrix:
    latent_matrix.shape  # (26694,200)

    # creating a collaborative latent matrix from user ratings:
    ratings_f1 = pd.merge(movies[['movieId']], ratings_f, on="movieId", how="right")

    # print("before : ", movies.__dict__)
    # print("after : ", movies.__dict__)
    print("ref count : ", sys.getrefcount(movies))

    print("ref count : ", sys.getrefcount(movies))
    '''
    def delete_me(obj):
        referrers = gc.get_referrers(obj)
        for referrer in referrers:
            if type(referrer) == dict:
                for key, value in referrer.items():
                    if value is obj:
                        referrer[key] = None
    delete_me(movies)
    '''
    movies = None
    tags = None
    ratings = None
    print("ref count : ", sys.getrefcount(movies))

    print("ref count : ", sys.getrefcount(movies))
    gc.collect()

    ratings_f2 = ratings_f1.pivot(index='movieId', columns='userId', values='rating').fillna(0)

    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=200)
    latent_matrix_2 = svd.fit_transform(ratings_f2)
    latent_matrix_2_df = pd.DataFrame(
        latent_matrix_2,
        index=Final.title.tolist())


    # plot variance expalined to see what latent dimensions to use
    #explained = svd.explained_variance_ratio_.cumsum()
    #plt.plot(explained, '.-', ms=16, color='red')
    #plt.xlabel('Singular value components', fontsize=12)
    #plt.ylabel('Cumulative percent of variance', fontsize=12)
    #plt.show()

    from sklearn.metrics.pairwise import cosine_similarity
    # take the latent vectors for a selected movie from both content
    # and collaborative matrices
    a_1 = np.array(latent_matrix_1_df.loc['Strada, La (1954)']).reshape(1, -1)
    a_2 = np.array(latent_matrix_2_df.loc['Strada, La (1954)']).reshape(1, -1)

    # calculate the similarity of this movie with the others in the list
    score_1 = cosine_similarity(latent_matrix_1_df, a_1).reshape(-1)
    score_2 = cosine_similarity(latent_matrix_2_df, a_2).reshape(-1)

    # an average measure of both content and collaborative
    hybrid = ((score_1 + score_2) / 2.0)

    # form a data frame of similar movies
    dictDf = {'content': score_1, 'collaborative': score_2, 'hybrid': hybrid}
    similar = pd.DataFrame(dictDf, index=latent_matrix_1_df.index)

    # sort it on the basis of either: content, collaborative or hybrid,
    # here : content
    similar.sort_values('content', ascending=False, inplace=True)
    print(similar[1:].head(11))



if __name__ == "__main__":
    # timetodate()
    # datashow(13)
    # naver_show(1)
    # movie_genre()
    movie_suggest()
    # temp1()
    # temp2()