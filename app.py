import requests
from flask import Flask, render_template, request
from requests.structures import CaseInsensitiveDict
import pandas as pd
import numpy as np
from prophet import Prophet
import json
import os

app = Flask(__name__)
app.config.from_object('config')

def get_token():
    url =os.environ["KEYCLOAK_BASE_URL"] + "/auth/realms/" + os.environ[
        "KEYCLOAK_REALM"] + "/protocol/openid-connect/token"

    print(url)
    headers = CaseInsensitiveDict()
    data = {"username":os.environ["TENANT_ADMIN_USERNAME"],
            "password":os.environ["TENANT_ADMIN_PASSWORD"],
            "grant_type": "password",
            "client_id": os.environ["KEYCLOAK_CLIENT_ID"]
            }
    headers["Accept"] = "application/json"
    response = requests.post(url, data=data, headers=headers)
    print(response);
    return response.json()['access_token']


def get_all_telemetries():
    url = os.environ["SERVER_BASE_URL"] + ":" + os.environ["BACKEND_PORT"] + "/api/plugins/telemetry/values/allTimeseries"
    print(url)
    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"
    headers["Authorization"] = "Bearer " + get_token()
    headers["user-agent"] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) ' \
                            'Chrome/86.0.4240.111 Safari/537.36 '

    response = requests.get(url, headers=headers, timeout=3)
    requests.adapters.DEFAULT_RETRIES = 5
    s = requests.session()
    s.keep_alive = False
    print(response.status_code)
    print()
    data_frame = pd.DataFrame.from_dict(response.json())
    print(data_frame)
    data_frame['doubleValue'] = data_frame['doubleValue'].fillna(data_frame['longValue'])
    print(data_frame.describe())
    data_frame = data_frame.dropna(subset=['doubleValue'])
    print(data_frame.describe())
    return data_frame


allTelemetry = get_all_telemetries()

def get_specific_telemetry(telemetry):
    dataframe_telemetry = allTelemetry.loc[allTelemetry['strKey'] == telemetry]
    dataframe_telemetry['ts'] = pd.to_datetime(dataframe_telemetry['ts'], unit='ms').dt.normalize()
    dataframe_telemetry = dataframe_telemetry.sort_values(by='ts')
    return dataframe_telemetry


def get_specific_telemetry_dateRange_calculatedAnalytics(telemetry, fromDate, toDate, entityId):
    if fromDate != "" and toDate != "" and entityId != "":
        dataframe_telemetry = allTelemetry.loc[allTelemetry['strKey'] == telemetry]
        dataframe_telemetry = dataframe_telemetry.loc[allTelemetry['entityId'] == entityId]

        dataframe_telemetry['ts'] = pd.to_datetime(dataframe_telemetry['ts'], unit='ms').dt.normalize()
        dataframe_telemetry = dataframe_telemetry.sort_values(by='ts')

        mask = (dataframe_telemetry['ts'] >= pd.to_datetime(fromDate, format='%d/%m/%Y')) & (
                    dataframe_telemetry['ts'] <= pd.to_datetime(toDate, format='%d/%m/%Y'))
        dataframe_telemetry = dataframe_telemetry.loc[mask]
        print(dataframe_telemetry.loc[mask])
        return dataframe_telemetry


    elif fromDate != "" and toDate != "" and entityId == "":
        dataframe_telemetry = allTelemetry.loc[allTelemetry['strKey'] == telemetry]
        dataframe_telemetry['ts'] = pd.to_datetime(dataframe_telemetry['ts'], unit='ms').dt.normalize()
        dataframe_telemetry = dataframe_telemetry.sort_values(by='ts')

        print('data before')
        print(dataframe_telemetry)
        print("pd.to_datetime(fromDate)")
        print(pd.to_datetime(fromDate, format='%d/%m/%Y'))

        mask = (dataframe_telemetry['ts'] >= pd.to_datetime(fromDate, format='%d/%m/%Y')) & (
                    dataframe_telemetry['ts'] <= pd.to_datetime(toDate, format='%d/%m/%Y'))
        dataframe_telemetry = dataframe_telemetry.loc[mask]
        print('data after')

        print(dataframe_telemetry.loc[mask])
        return dataframe_telemetry


    elif fromDate == "" and toDate == "" and entityId != "":
        dataframe_telemetry = allTelemetry.loc[allTelemetry['strKey'] == telemetry]
        dataframe_telemetry = dataframe_telemetry.loc[allTelemetry['entityId'] == entityId]

        dataframe_telemetry['ts'] = pd.to_datetime(dataframe_telemetry['ts'], unit='ms').dt.normalize()
        dataframe_telemetry = dataframe_telemetry.sort_values(by='ts')
        return dataframe_telemetry
    else:
        dataframe_telemetry = allTelemetry.loc[allTelemetry['strKey'] == telemetry]
        return dataframe_telemetry


def group_by_date_by_calculate_sum_for_each_day_withProphet(telemetry):
    dataframe_telemetry = get_specific_telemetry(telemetry)
    df_result = dataframe_telemetry.groupby('ts', as_index=False)['doubleValue'].sum().rename(
        columns={'ts': 'ds', 'doubleValue': 'y'})
    return df_result


def fill_with_previous_data_for_missing_dates(dataframe_telemetry):
    dataframe_telemetry = dataframe_telemetry.set_index('ds').asfreq('D').reset_index()
    dataframe_telemetry['y'].fillna(method='ffill', inplace=True)
    return dataframe_telemetry


def group_fill_acc(telemetryType):
    data_frame = group_by_date_by_calculate_sum_for_each_day_withProphet(telemetryType)
    data_frame = fill_with_previous_data_for_missing_dates(data_frame)
    data_frame['y'] = data_frame['y'].cumsum()
    return data_frame


@app.route('/')
def start():
    return "start"


@app.route('/api/AllTelemetries')
def show_all_telemetries():
    data_frame = allTelemetry
    return render_template('view.html', PageTitle="Pandas",
                           table=[data_frame.to_html(classes=["table "], index=False)],
                           titles=data_frame.columns.values)


@app.route('/api/specificTelemetries/<telemetryType>')
def show_specific_telemetries(telemetryType):
    data_frame = get_specific_telemetry(telemetryType)
    # return render_template('view.html', PageTitle="Pandas",
    #                        table=[data_frame.to_html(classes=["table"], index=False)],
    #                        titles=data_frame.columns.values)
    data_frame['ts'] = pd.to_datetime(data_frame['ts'], unit='ms').astype(str)
    return data_frame.to_json(orient='records')


@app.route('/api/prepareTelemetriesWithAcc/<telemetryType>')
def prepare_telemetries_with_prophet(telemetryType):
    data_frame = group_fill_acc(telemetryType)
    return render_template('view.html', PageTitle="Pandas",
                           table=[data_frame.to_html(classes=["table "], index=False)],
                           titles=data_frame.columns.values,
                           describeData=data_frame.describe())


# TODO : This route is running correctly when run app locally . but after we build a docker image from app it gave an
#  error related to prophet (the reason may be  version issue or os issue )
@app.route('/api/fit_predict_with_prophet/<telemetryType>')
def fit_predict_with_prophet(telemetryType):
    freq = request.args.get("freq", default="", type=str)
    periods = request.args.get("periods", default="", type=int)

    data_frame = group_fill_acc(telemetryType)
    model = Prophet()
    print(model.stan_backend)
    model.fit(data_frame)
    future_pd = model.make_future_dataframe(
        periods=periods, freq=freq, include_history=False
    )
    forecast_pd = model.predict(future_pd)
    print(forecast_pd[["ds", "yhat", 'yhat_lower', 'yhat_upper']])
    print(forecast_pd.describe())
    predict_fig = model.plot(forecast_pd, xlabel='date', ylabel='Energy')
    # plt.show()
    # return render_template('view.html', PageTitle="Pandas",
    #                        table=[future_pd.to_html(classes=["table "], index=False)],
    #                        titles=future_pd.columns.values,
    #                        )

    predicted = {
        'x': list(forecast_pd['ds'].astype(str)),
        'y': list(forecast_pd['yhat']),
    }

    return json.dumps(predicted)


@app.route('/api/calculated_analytics/<telemetryType>')
def calculated_analytics(telemetryType):
    fromDate = request.args.get("fromDate", default="", type=str)
    toDate = request.args.get("toDate", default="", type=str)
    entityId = request.args.get("entityId", default="", type=str)

    data_frame = get_specific_telemetry_dateRange_calculatedAnalytics(telemetryType, fromDate, toDate, entityId)
    print(data_frame.describe())
    # return render_template('view.html', PageTitle="Pandas",
    #                 table=[data_frame.to_html(classes=["table "], index=False)],
    #                 titles=data_frame.columns.values,
    #                 )

    analytics = {
        'count': float(data_frame['doubleValue'].count()),
        'mean': round(float(data_frame['doubleValue'].mean()), 6),
        'std': round(float(data_frame['doubleValue'].std()), 6),
        'min': float(data_frame['doubleValue'].min()),
        'max': float(data_frame['doubleValue'].max()),
        'sum': float(data_frame['doubleValue'].sum()),
    }

    analytics = {k: v if not np.isnan(v) else 0 for k, v in analytics.items()}
    print(analytics)
    # return data_frame.describe().to_json(orient='records')
    return json.dumps(analytics)


@app.route('/api/plot/<telemetryType>')
def plot(telemetryType):
    fromDate = request.args.get("fromDate", default="", type=str)
    toDate = request.args.get("toDate", default="", type=str)
    entityId = request.args.get("entityId", default="", type=str)

    data_frame = get_specific_telemetry_dateRange_calculatedAnalytics(telemetryType, fromDate, toDate, entityId)
    # return render_template('view.html', PageTitle="Pandas",
    #                 table=[data_frame.to_html(classes=["table "], index=False)],
    #                 titles=data_frame.columns.values,
    #                 )

    # r = sns.lineplot(data=data_frame, x=data_frame['ts'], y=data_frame['doubleValue'],
    #             )
    # r.set_xlabel("date")
    # r.set_ylabel(telemetryType)
    # bytes_image = io.BytesIO()
    # plt.savefig(bytes_image, format='png')
    # bytes_image.seek(0)
    data_frame = data_frame.groupby('ts', as_index=False)['doubleValue'].sum()
    plot = {
        'x': list(data_frame['ts'].astype(str)),
        'y': list(data_frame['doubleValue']),
    }

    return json.dumps(plot)


@app.route('/api/telemetriesNames')
def get_telemetries_names():
    telemetries = allTelemetry['strKey'].tolist()
    telemetriesNames = list(set(telemetries))
    return json.dumps(telemetriesNames)


if __name__ == '__main__':
    app.run(host="0.0.0.0")
