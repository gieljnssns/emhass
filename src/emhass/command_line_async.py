#!/usr/bin/env python3

import argparse
import asyncio
import copy
import logging
import os
import pathlib
import pickle
import re
from datetime import UTC, datetime
from importlib.metadata import version

import aiofiles
import numpy as np
import orjson
import pandas as pd

from emhass import utils_async as utils
from emhass.forecast_async import Forecast
from emhass.machine_learning_forecaster_async import MLForecaster
from emhass.machine_learning_regressor_async import MLRegressor
from emhass.optimization_async import Optimization
from emhass.retrieve_hass_async import RetrieveHass

default_csv_filename = "opt_res_latest.csv"
default_pkl_suffix = "_mlf.pkl"
default_metadata_json = "metadata.json"


async def retrieve_home_assistant_data(
    set_type: str,
    get_data_from_file: bool,
    retrieve_hass_conf: dict,
    optim_conf: dict,
    rh: RetrieveHass,
    emhass_conf: dict,
    test_df_literal: pd.DataFrame,
) -> dict:
    """Retrieve data from Home Assistant or file and prepare it for optimization."""
    if get_data_from_file:
        async with aiofiles.open(emhass_conf["data_path"] / test_df_literal, "rb") as inp:
            content = await inp.read()
            rh.df_final, days_list, var_list, rh.ha_config = pickle.loads(content)
            rh.var_list = var_list
        # Assign variables based on set_type
        retrieve_hass_conf["sensor_power_load_no_var_loads"] = str(var_list[0])
        if optim_conf.get("set_use_pv", True):
            retrieve_hass_conf["sensor_power_photovoltaics"] = str(var_list[1])
            retrieve_hass_conf["sensor_linear_interp"] = [
                retrieve_hass_conf["sensor_power_photovoltaics"],
                retrieve_hass_conf["sensor_power_load_no_var_loads"],
            ]
            retrieve_hass_conf["sensor_replace_zero"] = [
                retrieve_hass_conf["sensor_power_photovoltaics"],
                var_list[2],
            ]
        else:
            retrieve_hass_conf["sensor_linear_interp"] = [
                retrieve_hass_conf["sensor_power_load_no_var_loads"]
            ]
            retrieve_hass_conf["sensor_replace_zero"] = []
    else:
        # Determine days_list based on set_type
        if set_type == "perfect-optim" or set_type == "adjust_pv":
            days_list = utils.get_days_list(
                retrieve_hass_conf["historic_days_to_retrieve"]
            )
        elif set_type == "naive-mpc-optim":
            days_list = utils.get_days_list(1)
        else:
            days_list = None  # Not needed for dayahead
        var_list = [retrieve_hass_conf["sensor_power_load_no_var_loads"]]
        if optim_conf.get("set_use_pv", True):
            var_list.append(retrieve_hass_conf["sensor_power_photovoltaics"])
            if optim_conf.get("set_use_adjusted_pv", True):
                var_list.append(retrieve_hass_conf["sensor_power_photovoltaics_forecast"])
        if not await rh.get_data(
            days_list, var_list
        ):
            return False, None, days_list
    rh.prepare_data(
        retrieve_hass_conf["sensor_power_load_no_var_loads"],
        load_negative=retrieve_hass_conf["load_negative"],
        set_zero_min=retrieve_hass_conf["set_zero_min"],
        var_replace_zero=retrieve_hass_conf["sensor_replace_zero"],
        var_interp=retrieve_hass_conf["sensor_linear_interp"],
    )
    return True, rh.df_final.copy(), days_list


async def adjust_pv_forecast(
    logger: logging.Logger,
    fcst: Forecast,
    P_PV_forecast: pd.Series,
    get_data_from_file: bool,
    retrieve_hass_conf: dict,
    optim_conf: dict,
    rh: RetrieveHass,
    emhass_conf: dict,
    test_df_literal: pd.DataFrame,
) -> pd.Series:
    """
    Adjust the photovoltaic (PV) forecast using historical data and a regression model.

    This method retrieves historical data, prepares it for model fitting, trains a regression
    model, and adjusts the provided PV forecast based on the trained model.

    :param logger: Logger object for logging information and errors.
    :type logger: logging.Logger
    :param fcst: Forecast object used for PV forecast adjustment.
    :type fcst: Forecast
    :param P_PV_forecast: The initial PV forecast to be adjusted.
    :type P_PV_forecast: pd.Series
    :param get_data_from_file: Whether to retrieve data from a file instead of Home Assistant.
    :type get_data_from_file: bool
    :param retrieve_hass_conf: Configuration dictionary for retrieving data from Home Assistant.
    :type retrieve_hass_conf: dict
    :param optim_conf: Configuration dictionary for optimization settings.
    :type optim_conf: dict
    :param rh: RetrieveHass object for interacting with Home Assistant.
    :type rh: RetrieveHass
    :param emhass_conf: Configuration dictionary for emhass paths and settings.
    :type emhass_conf: dict
    :param test_df_literal: DataFrame containing test data for debugging purposes.
    :type test_df_literal: pd.DataFrame
    :return: The adjusted PV forecast as a pandas Series.
    :rtype: pd.Series
    """
    logger.info("Adjusting PV forecast, retrieving history data for model fit")
    # Retrieve data from Home Assistant
    success, df_input_data, _ = await retrieve_home_assistant_data(
        "adjust_pv",
        get_data_from_file,
        retrieve_hass_conf,
        optim_conf,
        rh,
        emhass_conf,
        test_df_literal,
    )
    if not success:
        return False
    # Call data preparation method
    await fcst.adjust_pv_forecast_data_prep(df_input_data)
    # Call the fit method
    await fcst.adjust_pv_forecast_fit(
        n_splits=5,
        regression_model=optim_conf["adjusted_pv_regression_model"],
    )
    # Call the predict method
    P_PV_forecast = P_PV_forecast.rename("forecast").to_frame()
    P_PV_forecast = await fcst.adjust_pv_forecast_predict(forecasted_pv=P_PV_forecast)
    # Update the PV forecast
    return P_PV_forecast["adjusted_forecast"].rename(None)


async def set_input_data_dict(
    emhass_conf: dict,
    costfun: str,
    params: str,
    runtimeparams: str,
    set_type: str,
    logger: logging.Logger,
    get_data_from_file: bool | None = False,
) -> dict:
    """
    Set up some of the data needed for the different actions.

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param costfun: The type of cost function to use for optimization problem
    :type costfun: str
    :param params: Configuration parameters passed from data/options.json
    :type params: str
    :param runtimeparams: Runtime optimization parameters passed as a dictionary
    :type runtimeparams: str
    :param set_type: Set the type of setup based on following type of optimization
    :type set_type: str
    :param logger: The passed logger object
    :type logger: logging object
    :param get_data_from_file: Use data from saved CSV file (useful for debug)
    :type get_data_from_file: bool, optional
    :return: A dictionnary with multiple data used by the action functions
    :rtype: dict

    """
    logger.info("Setting up needed data")

    # check if passed params is a dict
    if (params is not None) and (params != "null"):
        if type(params) is str:
            params = dict(orjson.loads(params))
    else:
        params = {}

    # Parsing yaml
    retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)

    if type(retrieve_hass_conf) is bool:
        return False

    # Treat runtimeparams
    params, retrieve_hass_conf, optim_conf, plant_conf = await utils.treat_runtimeparams(
        runtimeparams,
        params,
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        set_type,
        logger,
        emhass_conf,
    )

    # Define the data retrieve object
    rh = RetrieveHass(
        retrieve_hass_conf["hass_url"],
        retrieve_hass_conf["long_lived_token"],
        retrieve_hass_conf["optimization_time_step"],
        retrieve_hass_conf["time_zone"],
        params,
        emhass_conf,
        logger,
        get_data_from_file=get_data_from_file,
    )

    # Retrieve basic configuration data from hass
    test_df_literal = "test_df_final.pkl"
    if get_data_from_file:
        async with aiofiles.open(emhass_conf["data_path"] / test_df_literal, "rb") as inp:
            content = await inp.read()
            _, _, _, rh.ha_config = pickle.loads(content)
    else:
        response = await rh.get_ha_config()
        if type(response) is bool:
            return False

    # Update the params dict using data from the HA configuration
    params = utils.update_params_with_ha_config(
        params,
        rh.ha_config,
    )

    # Define the forecast and optimization objects
    fcst = Forecast(
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        params,
        emhass_conf,
        logger,
        get_data_from_file=get_data_from_file,
    )
    opt = Optimization(
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        fcst.var_load_cost,
        fcst.var_prod_price,
        costfun,
        emhass_conf,
        logger,
    )

    # Perform setup based on type of action
    if set_type == "perfect-optim":
        # Retrieve data from hass
        success, df_input_data, days_list = await retrieve_home_assistant_data(
            set_type,
            get_data_from_file,
            retrieve_hass_conf,
            optim_conf,
            rh,
            emhass_conf,
            test_df_literal,
        )
        if not success:
            return False
        # What we don't need for this type of action
        P_PV_forecast, P_load_forecast, df_input_data_dayahead = None, None, None
    elif set_type == "dayahead-optim":
        # Get PV and load forecasts
        if (
            optim_conf["set_use_pv"]
            or optim_conf.get("weather_forecast_method", None) == "list"
        ):
            df_weather = await fcst.get_weather_forecast(
                method=optim_conf["weather_forecast_method"]
            )
            if isinstance(df_weather, bool) and not df_weather:
                return False
            P_PV_forecast = fcst.get_power_from_weather(df_weather)
            # Adjust PV forecast
            if optim_conf["set_use_adjusted_pv"]:
                # Update the PV forecast
                P_PV_forecast = await adjust_pv_forecast(
                    logger,
                    fcst,
                    P_PV_forecast,
                    get_data_from_file,
                    retrieve_hass_conf,
                    optim_conf,
                    rh,
                    emhass_conf,
                    test_df_literal,
                )
        else:
            P_PV_forecast = pd.Series(0, index=fcst.forecast_dates)
        P_load_forecast = await fcst.get_load_forecast(
            days_min_load_forecast=optim_conf["delta_forecast_daily"].days,
            method=optim_conf["load_forecast_method"]
        )
        if isinstance(P_load_forecast, bool) and not P_load_forecast:
            logger.error(
                "Unable to get sensor power photovoltaics, or sensor power load no var loads. Check HA sensors and their daily data"
            )
            return False
        df_input_data_dayahead = pd.DataFrame(
            np.transpose(np.vstack([P_PV_forecast.values, P_load_forecast.values])),
            index=P_PV_forecast.index,
            columns=["P_PV_forecast", "P_load_forecast"],
        )
        if (
            "optimization_time_step" in retrieve_hass_conf
            and retrieve_hass_conf["optimization_time_step"]
        ):
            if not isinstance(
                retrieve_hass_conf["optimization_time_step"],
                pd._libs.tslibs.timedeltas.Timedelta,
            ):
                optimization_time_step = pd.to_timedelta(
                    retrieve_hass_conf["optimization_time_step"], "minute"
                )
            else:
                optimization_time_step = retrieve_hass_conf["optimization_time_step"]
            df_input_data_dayahead = df_input_data_dayahead.asfreq(
                optimization_time_step
            )
        else:
            df_input_data_dayahead = utils.set_df_index_freq(df_input_data_dayahead)
        params = orjson.loads(params)
        if (
            "prediction_horizon" in params["passed_data"]
            and params["passed_data"]["prediction_horizon"] is not None
        ):
            prediction_horizon = params["passed_data"]["prediction_horizon"]
            df_input_data_dayahead = copy.deepcopy(df_input_data_dayahead)[
                df_input_data_dayahead.index[0] : df_input_data_dayahead.index[
                    prediction_horizon - 1
                ]
            ]
        # What we don't need for this type of action
        df_input_data, days_list = None, None
    elif set_type == "naive-mpc-optim":
        if (
            optim_conf.get("load_forecast_method", None) == "list"
            and optim_conf.get("weather_forecast_method", None) == "list"
        ) or (
            optim_conf.get("load_forecast_method", None) == "list"
            and not (optim_conf["set_use_pv"])
        ):
            days_list = None
            set_mix_forecast = False
            df_input_data = None
        else:
            # Retrieve data from hass
            success, df_input_data, days_list = await retrieve_home_assistant_data(
                set_type,
                get_data_from_file,
                retrieve_hass_conf,
                optim_conf,
                rh,
                emhass_conf,
                test_df_literal,
            )
            if not success:
                return False
            set_mix_forecast = True
        # Get PV and load forecasts
        if (
            optim_conf["set_use_pv"]
            or optim_conf.get("weather_forecast_method", None) == "list"
        ):
            df_weather = await fcst.get_weather_forecast(
                method=optim_conf["weather_forecast_method"]
            )
            if isinstance(df_weather, bool) and not df_weather:
                return False
            P_PV_forecast = fcst.get_power_from_weather(
                df_weather, set_mix_forecast=set_mix_forecast, df_now=df_input_data
            )
            # Adjust PV forecast
            if optim_conf["set_use_adjusted_pv"]:
                # Update the PV forecast
                P_PV_forecast = await adjust_pv_forecast(
                    logger,
                    fcst,
                    P_PV_forecast,
                    get_data_from_file,
                    retrieve_hass_conf,
                    optim_conf,
                    rh,
                    emhass_conf,
                    test_df_literal,
                )
        else:
            P_PV_forecast = pd.Series(0, index=fcst.forecast_dates)
        P_load_forecast = await fcst.get_load_forecast(
            days_min_load_forecast=optim_conf["delta_forecast_daily"].days,
            method=optim_conf["load_forecast_method"],
            set_mix_forecast=set_mix_forecast,
            df_now=df_input_data,
        )
        if isinstance(P_load_forecast, bool) and not P_load_forecast:
            logger.error(
                "Unable to get sensor power photovoltaics, or sensor power load no var loads. Check HA sensors and their daily data"
            )
            return False

        df_input_data_dayahead = pd.concat([P_PV_forecast, P_load_forecast], axis=1)
        if (
            "optimization_time_step" in retrieve_hass_conf
            and retrieve_hass_conf["optimization_time_step"]
        ):
            if not isinstance(
                retrieve_hass_conf["optimization_time_step"],
                pd._libs.tslibs.timedeltas.Timedelta,
            ):
                optimization_time_step = pd.to_timedelta(
                    retrieve_hass_conf["optimization_time_step"], "minute"
                )
            else:
                optimization_time_step = retrieve_hass_conf["optimization_time_step"]
            df_input_data_dayahead = df_input_data_dayahead.asfreq(
                optimization_time_step
            )
        else:
            df_input_data_dayahead = utils.set_df_index_freq(df_input_data_dayahead)
        df_input_data_dayahead.columns = ["P_PV_forecast", "P_load_forecast"]
        params = orjson.loads(params)
        if (
            "prediction_horizon" in params["passed_data"]
            and params["passed_data"]["prediction_horizon"] is not None
        ):
            prediction_horizon = params["passed_data"]["prediction_horizon"]
            df_input_data_dayahead = copy.deepcopy(df_input_data_dayahead)[
                df_input_data_dayahead.index[0] : df_input_data_dayahead.index[
                    prediction_horizon - 1
                ]
            ]

    elif (
        set_type == "forecast-model-fit"
        or set_type == "forecast-model-predict"
        or set_type == "forecast-model-tune"
    ):
        df_input_data_dayahead = None
        P_PV_forecast, P_load_forecast = None, None
        params = orjson.loads(params)
        # Retrieve data from hass
        days_to_retrieve = params["passed_data"]["historic_days_to_retrieve"]
        model_type = params["passed_data"]["model_type"]
        var_model = params["passed_data"]["var_model"]
        if get_data_from_file:
            days_list = None
            filename = model_type + ".pkl"
            filename_path = emhass_conf["data_path"] / filename
            async with aiofiles.open(filename_path, "rb") as inp:
                content = await inp.read()
                df_input_data, _, _, _ = pickle.loads(content)
            df_input_data = df_input_data[
                df_input_data.index[-1] - pd.offsets.Day(days_to_retrieve) :
            ]
        else:
            days_list = utils.get_days_list(days_to_retrieve)
            var_list = [var_model]
            if not await rh.get_data(days_list, var_list):
                return False
            df_input_data = rh.df_final.copy()
    elif set_type == "regressor-model-fit" or set_type == "regressor-model-predict":
        df_input_data, df_input_data_dayahead = None, None
        P_PV_forecast, P_load_forecast = None, None
        params = orjson.loads(params)
        days_list = None
        csv_file = params["passed_data"].get("csv_file", None)
        if "features" in params["passed_data"]:
            features = params["passed_data"]["features"]
        if "target" in params["passed_data"]:
            target = params["passed_data"]["target"]
        if "timestamp" in params["passed_data"]:
            timestamp = params["passed_data"]["timestamp"]
        if csv_file:
            if get_data_from_file:
                base_path = emhass_conf["data_path"]  # + "/data"
                filename_path = pathlib.Path(base_path) / csv_file
            else:
                filename_path = emhass_conf["data_path"] / csv_file
            if filename_path.is_file():
                df_input_data = pd.read_csv(filename_path, parse_dates=True)
            else:
                logger.error(
                    "The CSV file "
                    + csv_file
                    + " was not found in path: "
                    + str(emhass_conf["data_path"])
                )
                return False
            required_columns = []
            required_columns.extend(features)
            required_columns.append(target)
            if timestamp is not None:
                required_columns.append(timestamp)
            if not set(required_columns).issubset(df_input_data.columns):
                logger.error("The cvs file does not contain the required columns.")
                msg = f"CSV file should contain the following columns: {', '.join(required_columns)}"
                logger.error(msg)
                return False
    elif set_type == "teach-deferrable":
        df_input_data, df_input_data_dayahead = None, None
        P_PV_forecast, P_load_forecast = None, None
        days_list = None
        params = orjson.loads(params)
        if "start_time" in params["passed_data"]:
            start_time = params["passed_data"]["start_time"]
        if "end_time" in params["passed_data"]:
            end_time = params["passed_data"]["end_time"]
        if "deferrable_load_id" in params["passed_data"]:
            deferrable_load_id = params["passed_data"]["deferrable_load_id"]
        if "profile_name" in params["passed_data"]:
            profile_name = params["passed_data"]["profile_name"]
    elif set_type == "publish-data":
        df_input_data, df_input_data_dayahead = None, None
        P_PV_forecast, P_load_forecast = None, None
        days_list = None
    else:
        logger.error(
            "The passed action argument and hence the set_type parameter for setup is not valid",
        )
        df_input_data, df_input_data_dayahead = None, None
        P_PV_forecast, P_load_forecast = None, None
        days_list = None
    # The input data dictionary to return
    input_data_dict = {
        "emhass_conf": emhass_conf,
        "retrieve_hass_conf": retrieve_hass_conf,
        "rh": rh,
        "opt": opt,
        "fcst": fcst,
        "df_input_data": df_input_data,
        "df_input_data_dayahead": df_input_data_dayahead,
        "P_PV_forecast": P_PV_forecast,
        "P_load_forecast": P_load_forecast,
        "costfun": costfun,
        "params": params,
        "days_list": days_list,
    }
    return input_data_dict


async def weather_forecast_cache(
    emhass_conf: dict, params: str, runtimeparams: str, logger: logging.Logger
) -> bool:
    """
    Perform a call to get forecast function, intend to save results to cache.

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param params: Configuration parameters passed from data/options.json
    :type params: str
    :param runtimeparams: Runtime optimization parameters passed as a dictionary
    :type runtimeparams: str
    :param logger: The passed logger object
    :type logger: logging object
    :return: A bool for function completion
    :rtype: bool

    """
    # Parsing yaml
    retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)
    # Treat runtimeparams
    params, retrieve_hass_conf, optim_conf, plant_conf = await utils.treat_runtimeparams(
        runtimeparams,
        params,
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        "forecast",
        logger,
        emhass_conf,
    )
    # Make sure weather_forecast_cache is true
    if (params is not None) and (params != "null"):
        params = orjson.loads(params)
    else:
        params = {}
    params["passed_data"]["weather_forecast_cache"] = True
    params = orjson.dumps(params).decode("utf-8")
    # Create Forecast object
    fcst = Forecast(
        retrieve_hass_conf, optim_conf, plant_conf, params, emhass_conf, logger
    )
    result = await fcst.get_weather_forecast(optim_conf["weather_forecast_method"])
    if isinstance(result, bool) and not result:
        return False

    return True


async def perfect_forecast_optim(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = True,
    debug: bool | None = False,
) -> pd.DataFrame:
    """
    Perform a call to the perfect forecast optimization routine.

    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: Save optimization results to CSV file
    :type save_data_to_file: bool, optional
    :param debug: A debug option useful for unittests
    :type debug: bool, optional
    :return: The output data of the optimization
    :rtype: pd.DataFrame

    """
    logger.info("Performing perfect forecast optimization")
    # Load cost and prod price forecast
    df_input_data = input_data_dict["fcst"].get_load_cost_forecast(
        input_data_dict["df_input_data"],
        method=input_data_dict["fcst"].optim_conf["load_cost_forecast_method"],
        list_and_perfect=True,
    )
    if isinstance(df_input_data, bool) and not df_input_data:
        return False
    df_input_data = input_data_dict["fcst"].get_prod_price_forecast(
        df_input_data,
        method=input_data_dict["fcst"].optim_conf["production_price_forecast_method"],
        list_and_perfect=True,
    )
    if isinstance(df_input_data, bool) and not df_input_data:
        return False

    # Load deferrable profiles if specified
    def_profile = None
    if "def_profile" in input_data_dict["params"]["passed_data"] and input_data_dict["params"]["passed_data"]["def_profile"] != None:
        profile_names = input_data_dict["params"]["passed_data"]["def_profile"]
        logger.info(f"def_profile parameter found: {profile_names}")

        # Check if any valid profile names are provided (not None, not empty string)
        valid_profiles = [name for name in profile_names if name and name != ""]
        if valid_profiles:
            logger.info(f"Loading {len(valid_profiles)} deferrable profiles: {valid_profiles}")
            def_profile = await load_deferrable_profiles(
                profile_names, input_data_dict["emhass_conf"], logger
            )
            if def_profile:
                logger.info(f"Successfully loaded {len([p for p in def_profile if p is not None])} profiles")
            else:
                logger.warning("No profiles were successfully loaded")
        else:
            logger.info("No valid profile names provided in def_profile parameter")
    else:
        logger.debug("No def_profile parameter found in runtime data")

    opt_res = await input_data_dict["opt"].perform_perfect_forecast_optim(
        df_input_data, input_data_dict["days_list"], def_profile=def_profile
    )
    # Save CSV file for analysis
    if save_data_to_file:
        filename = "opt_res_perfect_optim_" + input_data_dict["costfun"] + ".csv"
    else:  # Just save the latest optimization results
        filename = default_csv_filename
    if not debug:
        opt_res.to_csv(
            input_data_dict["emhass_conf"]["data_path"] / filename,
            index_label="timestamp",
        )
    if not isinstance(input_data_dict["params"], dict):
        params = orjson.loads(input_data_dict["params"])
    else:
        params = input_data_dict["params"]

    # if continual_publish, save perfect results to data_path/entities json
    if input_data_dict["retrieve_hass_conf"].get("continual_publish", False) or params[
        "passed_data"
    ].get("entity_save", False):
        # Trigger the publish function, save entity data and not post to HA
        await publish_data(input_data_dict, logger, entity_save=True, dont_post=True)

    return opt_res


async def dayahead_forecast_optim(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = False,
    debug: bool | None = False,
) -> pd.DataFrame:
    """
    Perform a call to the day-ahead optimization routine.

    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: Save optimization results to CSV file
    :type save_data_to_file: bool, optional
    :param debug: A debug option useful for unittests
    :type debug: bool, optional
    :return: The output data of the optimization
    :rtype: pd.DataFrame

    """
    logger.info("Performing day-ahead forecast optimization")
    # Load cost and prod price forecast
    df_input_data_dayahead = input_data_dict["fcst"].get_load_cost_forecast(
        input_data_dict["df_input_data_dayahead"],
        method=input_data_dict["fcst"].optim_conf["load_cost_forecast_method"],
    )
    if isinstance(df_input_data_dayahead, bool) and not df_input_data_dayahead:
        return False
    df_input_data_dayahead = input_data_dict["fcst"].get_prod_price_forecast(
        df_input_data_dayahead,
        method=input_data_dict["fcst"].optim_conf["production_price_forecast_method"],
    )
    if isinstance(df_input_data_dayahead, bool) and not df_input_data_dayahead:
        return False
    if "outdoor_temperature_forecast" in input_data_dict["params"]["passed_data"]:
        df_input_data_dayahead["outdoor_temperature_forecast"] = input_data_dict[
            "params"
        ]["passed_data"]["outdoor_temperature_forecast"]

    # Load deferrable profiles if specified
    def_profile = None
    if "def_profile" in input_data_dict["params"]["passed_data"] and input_data_dict["params"]["passed_data"]["def_profile"] != None:
        profile_names = input_data_dict["params"]["passed_data"]["def_profile"]
        logger.info(f"def_profile parameter found: {profile_names}")

        # Check if any valid profile names are provided (not None, not empty string)
        valid_profiles = [name for name in profile_names if name and name != ""]
        if valid_profiles:
            logger.info(f"Loading {len(valid_profiles)} deferrable profiles: {valid_profiles}")
            def_profile = await load_deferrable_profiles(
                profile_names, input_data_dict["emhass_conf"], logger
            )
            if def_profile:
                logger.info(f"Successfully loaded {len([p for p in def_profile if p is not None])} profiles")
            else:
                logger.warning("No profiles were successfully loaded")
        else:
            logger.info("No valid profile names provided in def_profile parameter")
    else:
        logger.debug("No def_profile parameter found in runtime data")

    opt_res_dayahead = await input_data_dict["opt"].perform_dayahead_forecast_optim(
        df_input_data_dayahead,
        input_data_dict["P_PV_forecast"],
        input_data_dict["P_load_forecast"],
        def_profile=def_profile,
    )
    # Save CSV file for publish_data
    if save_data_to_file:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = "opt_res_dayahead_" + today.strftime("%Y_%m_%d") + ".csv"
    else:  # Just save the latest optimization results
        filename = default_csv_filename
    if not debug:
        opt_res_dayahead.to_csv(
            input_data_dict["emhass_conf"]["data_path"] / filename,
            index_label="timestamp",
        )

    if not isinstance(input_data_dict["params"], dict):
        params = orjson.loads(input_data_dict["params"])
    else:
        params = input_data_dict["params"]

    # if continual_publish, save day_ahead results to data_path/entities json
    if input_data_dict["retrieve_hass_conf"].get("continual_publish", False) or params[
        "passed_data"
    ].get("entity_save", False):
        # Trigger the publish function, save entity data and not post to HA
        await publish_data(input_data_dict, logger, entity_save=True, dont_post=True)

    return opt_res_dayahead


async def naive_mpc_optim(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = False,
    debug: bool | None = False,
) -> pd.DataFrame:
    """
    Perform a call to the naive Model Predictive Controller optimization routine.

    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: Save optimization results to CSV file
    :type save_data_to_file: bool, optional
    :param debug: A debug option useful for unittests
    :type debug: bool, optional
    :return: The output data of the optimization
    :rtype: pd.DataFrame

    """
    logger.info("Performing naive MPC optimization")

    # Load cost and prod price forecast
    df_input_data_dayahead = input_data_dict["fcst"].get_load_cost_forecast(
        input_data_dict["df_input_data_dayahead"],
        method=input_data_dict["fcst"].optim_conf["load_cost_forecast_method"],
    )
    if isinstance(df_input_data_dayahead, bool) and not df_input_data_dayahead:
        return False
    df_input_data_dayahead = input_data_dict["fcst"].get_prod_price_forecast(
        df_input_data_dayahead,
        method=input_data_dict["fcst"].optim_conf["production_price_forecast_method"],
    )
    if isinstance(df_input_data_dayahead, bool) and not df_input_data_dayahead:
        return False
    if "outdoor_temperature_forecast" in input_data_dict["params"]["passed_data"]:
        df_input_data_dayahead["outdoor_temperature_forecast"] = input_data_dict[
            "params"
        ]["passed_data"]["outdoor_temperature_forecast"]
    # The specifics params for the MPC at runtime
    prediction_horizon = input_data_dict["params"]["passed_data"]["prediction_horizon"]
    soc_init = input_data_dict["params"]["passed_data"]["soc_init"]
    soc_final = input_data_dict["params"]["passed_data"]["soc_final"]
    def_total_hours = input_data_dict["params"]["optim_conf"].get(
        "operating_hours_of_each_deferrable_load", None
    )
    def_total_timestep = input_data_dict["params"]["optim_conf"].get(
        "operating_timesteps_of_each_deferrable_load", None
    )
    def_start_timestep = input_data_dict["params"]["optim_conf"][
        "start_timesteps_of_each_deferrable_load"
    ]
    def_end_timestep = input_data_dict["params"]["optim_conf"][
        "end_timesteps_of_each_deferrable_load"
    ]

    # Load deferrable profiles if specified
    def_profile = None
    if "def_profile" in input_data_dict["params"]["passed_data"] and input_data_dict["params"]["passed_data"]["def_profile"] != None:
        profile_names = input_data_dict["params"]["passed_data"]["def_profile"]
        logger.info(f"def_profile parameter found: {profile_names}")

        # Check if any valid profile names are provided (not None, not empty string)
        valid_profiles = [name for name in profile_names if name and name != ""]
        if valid_profiles:
            logger.info(f"Loading {len(valid_profiles)} deferrable profiles: {valid_profiles}")
            def_profile = await load_deferrable_profiles(
                profile_names, input_data_dict["emhass_conf"], logger
            )
            if def_profile:
                logger.info(f"Successfully loaded {len([p for p in def_profile if p is not None])} profiles")
            else:
                logger.warning("No profiles were successfully loaded")
        else:
            logger.info("No valid profile names provided in def_profile parameter")
    else:
        logger.debug("No def_profile parameter found in runtime data")


    opt_res_naive_mpc = await input_data_dict["opt"].perform_naive_mpc_optim(
        df_input_data_dayahead,
        input_data_dict["P_PV_forecast"],
        input_data_dict["P_load_forecast"],
        prediction_horizon,
        soc_init,
        soc_final,
        def_total_hours,
        def_total_timestep,
        def_start_timestep,
        def_end_timestep,
        def_profile=def_profile,
    )
    # Save CSV file for publish_data
    if save_data_to_file:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = "opt_res_naive_mpc_" + today.strftime("%Y_%m_%d") + ".csv"
    else:  # Just save the latest optimization results
        filename = default_csv_filename
    if not debug:
        opt_res_naive_mpc.to_csv(
            input_data_dict["emhass_conf"]["data_path"] / filename,
            index_label="timestamp",
        )

    if not isinstance(input_data_dict["params"], dict):
        params = orjson.loads(input_data_dict["params"])
    else:
        params = input_data_dict["params"]

    # if continual_publish, save mpc results to data_path/entities json
    if input_data_dict["retrieve_hass_conf"].get("continual_publish", False) or params[
        "passed_data"
    ].get("entity_save", False):
        # Trigger the publish function, save entity data and not post to HA
        await publish_data(input_data_dict, logger, entity_save=True, dont_post=True)

    return opt_res_naive_mpc


async def forecast_model_fit(
    input_data_dict: dict, logger: logging.Logger, debug: bool | None = False
) -> tuple[pd.DataFrame, pd.DataFrame, MLForecaster]:
    """Perform a forecast model fit from training data retrieved from Home Assistant.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    :return: The DataFrame containing the forecast data results without and with backtest and the `mlforecaster` object
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, mlforecaster]
    """
    data = copy.deepcopy(input_data_dict["df_input_data"])
    model_type = input_data_dict["params"]["passed_data"]["model_type"]
    var_model = input_data_dict["params"]["passed_data"]["var_model"]
    sklearn_model = input_data_dict["params"]["passed_data"]["sklearn_model"]
    num_lags = input_data_dict["params"]["passed_data"]["num_lags"]
    split_date_delta = input_data_dict["params"]["passed_data"]["split_date_delta"]
    perform_backtest = input_data_dict["params"]["passed_data"]["perform_backtest"]
    # The ML forecaster object
    mlf = MLForecaster(
        data,
        model_type,
        var_model,
        sklearn_model,
        num_lags,
        input_data_dict["emhass_conf"],
        logger,
    )
    # Fit the ML model
    df_pred, df_pred_backtest = await mlf.fit(
        split_date_delta=split_date_delta, perform_backtest=perform_backtest
    )
    # Save model
    if not debug:
        filename = model_type + default_pkl_suffix
        filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
        async with aiofiles.open(filename_path, "wb") as outp:
            await outp.write(pickle.dumps(mlf, pickle.HIGHEST_PROTOCOL))
            logger.debug("saved model to " + str(filename_path))
    return df_pred, df_pred_backtest, mlf


async def forecast_model_predict(
    input_data_dict: dict,
    logger: logging.Logger,
    use_last_window: bool | None = True,
    debug: bool | None = False,
    mlf: MLForecaster | None = None,
) -> pd.DataFrame:
    r"""Perform a forecast model predict using a previously trained skforecast model.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param use_last_window: True if the 'last_window' option should be used for the \
        custom machine learning forecast model. The 'last_window=True' means that the data \
        that will be used to generate the new forecast will be freshly retrieved from \
        Home Assistant. This data is needed because the forecast model is an auto-regressive \
        model with lags. If 'False' then the data using during the model train is used. Defaults to True
    :type use_last_window: Optional[bool], optional
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    :param mlf: The 'mlforecaster' object previously trained. This is mainly used for debug \
        and unit testing. In production the actual model will be read from a saved pickle file. Defaults to None
    :type mlf: Optional[mlforecaster], optional
    :return: The DataFrame containing the forecast prediction data
    :rtype: pd.DataFrame
    """
    # Load model
    model_type = input_data_dict["params"]["passed_data"]["model_type"]
    filename = model_type + default_pkl_suffix
    filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
    if not debug:
        if filename_path.is_file():
            async with aiofiles.open(filename_path, "rb") as inp:
                content = await inp.read()
                mlf = pickle.loads(content)
                logger.debug("loaded saved model from " + str(filename_path))
        else:
            logger.error(
                "The ML forecaster file ("
                + str(filename_path)
                + ") was not found, please run a model fit method before this predict method",
            )
            return
    # Make predictions
    if use_last_window:
        data_last_window = copy.deepcopy(input_data_dict["df_input_data"])
    else:
        data_last_window = None
    predictions = await mlf.predict(data_last_window)
    # Publish data to a Home Assistant sensor
    model_predict_publish = input_data_dict["params"]["passed_data"][
        "model_predict_publish"
    ]
    model_predict_entity_id = input_data_dict["params"]["passed_data"][
        "model_predict_entity_id"
    ]
    model_predict_device_class = input_data_dict["params"]["passed_data"][
        "model_predict_device_class"
    ]
    model_predict_unit_of_measurement = input_data_dict["params"]["passed_data"][
        "model_predict_unit_of_measurement"
    ]
    model_predict_friendly_name = input_data_dict["params"]["passed_data"][
        "model_predict_friendly_name"
    ]
    publish_prefix = input_data_dict["params"]["passed_data"]["publish_prefix"]
    if model_predict_publish is True:
        # Estimate the current index
        now_precise = datetime.now(
            input_data_dict["retrieve_hass_conf"]["time_zone"]
        ).replace(second=0, microsecond=0)
        if input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "nearest":
            idx_closest = predictions.index.get_indexer(
                [now_precise], method="nearest"
            )[0]
        elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "first":
            idx_closest = predictions.index.get_indexer([now_precise], method="ffill")[
                0
            ]
        elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "last":
            idx_closest = predictions.index.get_indexer([now_precise], method="bfill")[
                0
            ]
        if idx_closest == -1:
            idx_closest = predictions.index.get_indexer(
                [now_precise], method="nearest"
            )[0]
        # Publish Load forecast
        await input_data_dict["rh"].post_data(
            predictions,
            idx_closest,
            model_predict_entity_id,
            model_predict_device_class,
            model_predict_unit_of_measurement,
            model_predict_friendly_name,
            type_var="mlforecaster",
            publish_prefix=publish_prefix,
        )
    return predictions


async def forecast_model_tune(
    input_data_dict: dict,
    logger: logging.Logger,
    debug: bool | None = False,
    mlf: MLForecaster | None = None,
) -> tuple[pd.DataFrame, MLForecaster]:
    """Tune a forecast model hyperparameters using bayesian optimization.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    :param mlf: The 'mlforecaster' object previously trained. This is mainly used for debug \
        and unit testing. In production the actual model will be read from a saved pickle file. Defaults to None
    :type mlf: Optional[mlforecaster], optional
    :return: The DataFrame containing the forecast data results using the optimized model
    :rtype: pd.DataFrame
    """
    # Load model
    model_type = input_data_dict["params"]["passed_data"]["model_type"]
    filename = model_type + default_pkl_suffix
    filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
    if not debug:
        if filename_path.is_file():
            async with aiofiles.open(filename_path, "rb") as inp:
                content = await inp.read()
                mlf = pickle.loads(content)
                logger.debug("loaded saved model from " + str(filename_path))
        else:
            logger.error(
                "The ML forecaster file ("
                + str(filename_path)
                + ") was not found, please run a model fit method before this tune method",
            )
            return None, None
    # Tune the model
    df_pred_optim = await mlf.tune(debug=debug)
    # Save model
    if not debug:
        filename = model_type + default_pkl_suffix
        filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
        async with aiofiles.open(filename_path, "wb") as outp:
            await outp.write(pickle.dumps(mlf, pickle.HIGHEST_PROTOCOL))
            logger.debug("Saved model to " + str(filename_path))
    return df_pred_optim, mlf


async def regressor_model_fit(
    input_data_dict: dict, logger: logging.Logger, debug: bool | None = False
) -> MLRegressor:
    """Perform a forecast model fit from training data retrieved from Home Assistant.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    """
    data = copy.deepcopy(input_data_dict["df_input_data"])
    if "model_type" in input_data_dict["params"]["passed_data"]:
        model_type = input_data_dict["params"]["passed_data"]["model_type"]
    else:
        logger.error("parameter: 'model_type' not passed")
        return False
    if "regression_model" in input_data_dict["params"]["passed_data"]:
        regression_model = input_data_dict["params"]["passed_data"]["regression_model"]
    else:
        logger.error("parameter: 'regression_model' not passed")
        return False
    if "features" in input_data_dict["params"]["passed_data"]:
        features = input_data_dict["params"]["passed_data"]["features"]
    else:
        logger.error("parameter: 'features' not passed")
        return False
    if "target" in input_data_dict["params"]["passed_data"]:
        target = input_data_dict["params"]["passed_data"]["target"]
    else:
        logger.error("parameter: 'target' not passed")
        return False
    if "timestamp" in input_data_dict["params"]["passed_data"]:
        timestamp = input_data_dict["params"]["passed_data"]["timestamp"]
    else:
        logger.error("parameter: 'timestamp' not passed")
        return False
    if "date_features" in input_data_dict["params"]["passed_data"]:
        date_features = input_data_dict["params"]["passed_data"]["date_features"]
    else:
        logger.error("parameter: 'date_features' not passed")
        return False
    # The MLRegressor object
    mlr = MLRegressor(
        data, model_type, regression_model, features, target, timestamp, logger
    )
    # Fit the ML model
    fit = await mlr.fit(date_features=date_features)
    if not fit:
        return False
    # Save model
    if not debug:
        filename = model_type + "_mlr.pkl"
        filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
        async with aiofiles.open(filename_path, "wb") as outp:
            await outp.write(pickle.dumps(mlr, pickle.HIGHEST_PROTOCOL))
    return mlr


async def regressor_model_predict(
    input_data_dict: dict,
    logger: logging.Logger,
    debug: bool | None = False,
    mlr: MLRegressor | None = None,
) -> np.ndarray:
    """Perform a prediction from csv file.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    """
    if "model_type" in input_data_dict["params"]["passed_data"]:
        model_type = input_data_dict["params"]["passed_data"]["model_type"]
    else:
        logger.error("parameter: 'model_type' not passed")
        return False
    filename = model_type + "_mlr.pkl"
    filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
    if not debug:
        if filename_path.is_file():
            async with aiofiles.open(filename_path, "rb") as inp:
                content = await inp.read()
                mlr = pickle.loads(content)
        else:
            logger.error(
                "The ML forecaster file was not found, please run a model fit method before this predict method",
            )
            return False
    if "new_values" in input_data_dict["params"]["passed_data"]:
        new_values = input_data_dict["params"]["passed_data"]["new_values"]
    else:
        logger.error("parameter: 'new_values' not passed")
        return False
    # Predict from csv file
    prediction = await mlr.predict(new_values)
    mlr_predict_entity_id = input_data_dict["params"]["passed_data"].get(
        "mlr_predict_entity_id", "sensor.mlr_predict"
    )
    mlr_predict_device_class = input_data_dict["params"]["passed_data"].get(
        "mlr_predict_device_class", "power"
    )
    mlr_predict_unit_of_measurement = input_data_dict["params"]["passed_data"].get(
        "mlr_predict_unit_of_measurement", "W"
    )
    mlr_predict_friendly_name = input_data_dict["params"]["passed_data"].get(
        "mlr_predict_friendly_name", "mlr predictor"
    )
    # Publish prediction
    idx = 0
    if not debug:
        await input_data_dict["rh"].post_data(
            prediction,
            idx,
            mlr_predict_entity_id,
            mlr_predict_device_class,
            mlr_predict_unit_of_measurement,
            mlr_predict_friendly_name,
            type_var="mlregressor",
        )
    return prediction


async def load_deferrable_profiles(
    profile_names: list,
    emhass_conf: dict,
    logger: logging.Logger,
) -> list:
    """
    Load deferrable load profiles from saved files.

    :param profile_names: List of profile names to load
    :type profile_names: list
    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :return: List of profile names that were successfully validated (None for profiles that couldn't be loaded)
    :rtype: list
    """
    logger.info(f"Validating deferrable profiles: {profile_names}")
    profiles = []
    profiles_dir = emhass_conf["data_path"] / "deferrable_profiles"
    logger.debug(f"Profiles directory: {profiles_dir}")

    for profile_name in profile_names:
        if profile_name is None or profile_name == "" or profile_name == 0:
            logger.debug(f"Skipping empty/None profile name")
            profiles.append(None)
            continue

        profile_filename = f"{profile_name}.pkl"
        profile_path = profiles_dir / profile_filename
        logger.debug(f"Looking for profile file: {profile_path}")

        try:
            if profile_path.exists():
                async with aiofiles.open(profile_path, "rb") as f:
                    content = await f.read()
                    profile_data = pickle.loads(content)
                    if "load_profile" in profile_data and profile_data["load_profile"]:
                        profiles.append(profile_name)
                        logger.info(f"Validated deferrable profile: {profile_name} with {len(profile_data['load_profile'])} data points")
                    else:
                        logger.warning(f"Invalid profile data in {profile_name}")
                        profiles.append(None)
            else:
                logger.warning(f"Deferrable profile not found: {profile_path}")
                profiles.append(None)
        except Exception as e:
            logger.error(f"Error validating deferrable profile {profile_name}: {str(e)}")
            profiles.append(None)

    logger.info(f"Finished validating profiles: {len([p for p in profiles if p is not None])} successful out of {len(profiles)} total")
    return profiles


async def debug_deferrable_status(
    input_data_dict: dict,
    logger: logging.Logger,
    debug: bool | None = False,
) -> dict:
    """
    Debug function to analyze current deferrable load status and profile data.

    This function helps diagnose issues with deferrable load optimization
    by checking current sensor values, available profiles, and optimization parameters.

    :param input_data_dict: A dictionary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    :return: Dictionary containing analysis results
    :rtype: dict
    """
    logger.info("Starting deferrable load status analysis")

    analysis_results = {
        "available_profiles": [],
        "sensor_data": {},
        "runtime_parameters": {},
        "recommendations": []
    }

    try:
        # Check runtime parameters
        passed_data = input_data_dict.get("params", {}).get("passed_data", {})
        analysis_results["runtime_parameters"] = {
            "def_profile": passed_data.get("def_profile", []),
            "p_deferrable_nom": passed_data.get("p_deferrable_nom", []),
            "def_total_hours": passed_data.get("def_total_hours", [])
        }

        # Check available profiles
        profiles_dir = input_data_dict["emhass_conf"]["data_path"] / "deferrable_profiles"
        if profiles_dir.exists():
            for profile_file in profiles_dir.glob("*.pkl"):
                try:
                    async with aiofiles.open(profile_file, "rb") as f:
                        content = await f.read()
                        profile_data = pickle.loads(content)

                    analysis_results["available_profiles"].append({
                        "name": profile_file.stem,
                        "deferrable_load_id": profile_data.get("deferrable_load_id", "unknown"),
                        "profile_length": len(profile_data.get("load_profile", [])),
                        "total_energy": sum(profile_data.get("load_profile", [])) *
                                      input_data_dict.get("retrieve_hass_conf", {}).get("optimization_time_step", 30) / 60,
                        "created_timestamp": profile_data.get("created_timestamp", "unknown")
                    })
                except Exception as e:
                    logger.warning(f"Could not analyze profile {profile_file}: {e}")

        # Check current sensor data
        rh = input_data_dict.get("rh")
        if rh and hasattr(rh, 'df_final') and rh.df_final is not None:
            df = rh.df_final.copy()
            analysis_results["sensor_data"]["available_columns"] = list(df.columns)
            analysis_results["sensor_data"]["data_range"] = {
                "start": str(df.index[0]) if len(df) > 0 else "No data",
                "end": str(df.index[-1]) if len(df) > 0 else "No data",
                "length": len(df)
            }

            # Analyze potential deferrable load sensors
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['deferrable', 'wash', 'dryer', 'dishwash', 'heat_pump']):
                    recent_values = df[col].iloc[:5].values if len(df) >= 5 else df[col].values
                    analysis_results["sensor_data"][col] = {
                        "recent_values": recent_values.tolist(),
                        "avg_recent": float(recent_values.mean()) if len(recent_values) > 0 else 0,
                        "max_recent": float(recent_values.max()) if len(recent_values) > 0 else 0
                    }

        # Generate recommendations
        def_profile = analysis_results["runtime_parameters"]["def_profile"]
        if def_profile and any(p is not None for p in def_profile):
            available_names = [p["name"] for p in analysis_results["available_profiles"]]
            for i, profile_name in enumerate(def_profile):
                if profile_name and profile_name not in available_names:
                    analysis_results["recommendations"].append(
                        f"Profile '{profile_name}' for load {i} not found. Available: {available_names}"
                    )
                elif profile_name in available_names:
                    profile_info = next(p for p in analysis_results["available_profiles"] if p["name"] == profile_name)
                    sensor_id = profile_info["deferrable_load_id"]
                    if sensor_id not in analysis_results["sensor_data"]:
                        analysis_results["recommendations"].append(
                            f"Sensor '{sensor_id}' for profile '{profile_name}' not found in current data"
                        )

        if not analysis_results["available_profiles"]:
            analysis_results["recommendations"].append(
                "No deferrable profiles found. Use /action/teach-deferrable to create profiles first."
            )

        logger.info(f"Analysis complete. Found {len(analysis_results['available_profiles'])} profiles, {len(analysis_results['recommendations'])} recommendations")
        return analysis_results

    except Exception as e:
        logger.error(f"Error in debug_deferrable_status: {str(e)}")
        analysis_results["error"] = str(e)
        return analysis_results


async def teach_deferrable(
    input_data_dict: dict,
    logger: logging.Logger,
    debug: bool | None = False,
) -> bool:
    """
    Teach/learn a deferrable load profile from Home Assistant historical data.

    This function retrieves historical consumption data for a deferrable load
    between specified start and end hours, and saves the profile for later use
    in optimization.

    :param input_data_dict: A dictionary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    :return: True if successful, False otherwise
    :rtype: bool
    """
    logger.info("Starting deferrable load profile teaching")

    # Get required parameters from runtime data
    passed_data = input_data_dict["params"]["passed_data"]

    # Check for required parameters
    required_params = ["start_time", "end_time", "deferrable_load_id"]
    for param in required_params:
        if param not in passed_data:
            logger.error(f"Required parameter '{param}' not provided")
            return False

    start_time = passed_data["start_time"]
    end_time = passed_data["end_time"]
    deferrable_load_id = passed_data["deferrable_load_id"]

    # Optional parameters
    profile_name = passed_data.get("profile_name", f"profile_{deferrable_load_id}")
    # days_to_retrieve = passed_data.get("days_to_retrieve", 7)  # Default to 7 days
    # sensor_name = passed_data.get("sensor_name", deferrable_load_id)

    logger.info(f"Teaching deferrable load profile: {profile_name}")
    # logger.info(f"Sensor: {sensor_name}, Start: {start_time}, End: {end_time}, Days: {days_to_retrieve}")

    try:
        # Create a temporary retrieve_hass_conf for data retrieval
        retrieve_hass_conf = input_data_dict["params"]["retrieve_hass_conf"].copy()

        # Get historical data
        days_list = None
        var_list = [deferrable_load_id]
        start_time_teach = start_time
        end_time_teach = end_time

        # Use the existing RetrieveHass instance to get data
        rh = input_data_dict["rh"]

        if not await rh.get_data(days_list, var_list, start_time_teach, end_time_teach):
            logger.error("Failed to retrieve data from Home Assistant")
            return False

        # Prepare the data
        rh.prepare_data(
            deferrable_load_id,
            load_negative=retrieve_hass_conf.get("load_negative", True),
            set_zero_min=retrieve_hass_conf.get("set_zero_min", True),
            var_replace_zero=[],
            var_interp=[deferrable_load_id],
        )

        df = rh.df_final.copy()

        # Filter data to only include the specified time window
        # Convert start_time and end_time to time objects for filtering
        # print(df)
        # df_filtered = df[
        #     (df.index.hour >= start_time) & (df.index.hour < end_time)
        # ]

        if df.empty:
            logger.error(f"No data found between hours {start_time} and {end_time}")
            return False
        deferrable_load_id_new = deferrable_load_id + "_positive"
        # Extract the load profile values for the sensor
        if deferrable_load_id_new not in df.columns:
            logger.error(f"Sensor '{deferrable_load_id}' not found in retrieved data")
            return False
        load_values = df[deferrable_load_id_new].values

        # Calculate statistics
        # mean_power = np.mean(load_values)
        # max_power = np.max(df["max"].values)
        # min_power = np.min(df["min"].values)
        # total_energy = np.sum(load_values) * (retrieve_hass_conf["optimization_time_step"].seconds / 3600)  # kWh
        # Create profile data structure
        profile_data = {
            "profile_name": profile_name,
            "deferrable_load_id": deferrable_load_id_new,
            "start_time": start_time,
            "end_time": end_time,
            "created_timestamp": datetime.now(UTC).isoformat(),
            "load_profile": load_values.tolist(),
            "statistics": {
                # "mean_power": float(mean_power),
                # "max_power": float(max_power),
                # "min_power": float(min_power),
                # "total_energy": float(total_energy),
                "num_datapoints": len(load_values)
            },
            # "time_step_minutes": pd.to_timedelta(
            #     retrieve_hass_conf["optimization_time_step"], "minute"
            # )
        }

        # Save profile to file
        profiles_dir = input_data_dict["emhass_conf"]["data_path"] / "deferrable_profiles"
        profiles_dir.mkdir(exist_ok=True)

        profile_filename = f"{profile_name}.pkl"
        profile_path = profiles_dir / profile_filename

        if not debug:
            async with aiofiles.open(profile_path, "wb") as f:
                await f.write(pickle.dumps(profile_data))

        logger.info(f"Deferrable load profile saved to: {profile_path}")
        # logger.info(f"Profile statistics - Mean: {mean_power:.2f}W, Max: {max_power:.2f}W, Total Energy: {total_energy:.2f}kWh")

        return True

    except Exception as e:
        logger.error(f"Error in teach_deferrable: {str(e)}")
        return False


async def publish_data(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = False,
    opt_res_latest: pd.DataFrame | None = None,
    entity_save: bool | None = False,
    dont_post: bool | None = False,
) -> pd.DataFrame:
    """
    Publish the data obtained from the optimization results.

    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: If True we will read data from optimization results in dayahead CSV file
    :type save_data_to_file: bool, optional
    :return: The output data of the optimization readed from a CSV file in the data folder
    :rtype: pd.DataFrame
    :param entity_save: Save built entities to data_path/entities
    :type entity_save: bool, optional
    :param dont_post: Do not post to Home Assistant. Works with entity_save
    :type dont_post: bool, optional

    """
    logger.info("Publishing data to HASS instance")
    if input_data_dict:
        if not isinstance(input_data_dict.get("params", {}), dict):
            params = orjson.loads(input_data_dict["params"])
        else:
            params = input_data_dict.get("params", {})

    # Check if a day ahead optimization has been performed (read CSV file)
    if save_data_to_file:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = "opt_res_dayahead_" + today.strftime("%Y_%m_%d") + ".csv"
    # If publish_prefix is passed, check if there is saved entities in data_path/entities with prefix, publish to results
    elif params["passed_data"].get("publish_prefix", "") != "" and not dont_post:
        opt_res_list = []
        opt_res_list_names = []
        publish_prefix = params["passed_data"]["publish_prefix"]
        entity_path = input_data_dict["emhass_conf"]["data_path"] / "entities"
        # Check if items in entity_path
        if os.path.exists(entity_path) and len(os.listdir(entity_path)) > 0:
            # Obtain all files in entity_path
            entity_path_contents = os.listdir(entity_path)
            # Confirm the entity path contains at least one file containing publish prefix or publish_prefix='all'
            if (
                any(publish_prefix in entity for entity in entity_path_contents)
                or publish_prefix == "all"
            ):
                # Loop through all items in entity path
                for entity in entity_path_contents:
                    # If publish_prefix is "all" publish all saved entities to Home Assistant
                    # If publish_prefix matches the prefix from saved entities, publish to Home Assistant
                    if entity != default_metadata_json and (
                        publish_prefix in entity or publish_prefix == "all"
                    ):
                        entity_data = await publish_json(
                            entity, input_data_dict, entity_path, logger
                        )
                        if not isinstance(entity_data, bool):
                            opt_res_list.append(entity_data)
                            opt_res_list_names.append(entity.replace(".json", ""))
                        else:
                            return False
                # Build a DataFrame with published entities
                opt_res = pd.concat(opt_res_list, axis=1)
                opt_res.columns = opt_res_list_names
                return opt_res
            else:
                logger.warning(
                    "No saved entity json files that match prefix: "
                    + str(publish_prefix)
                )
                logger.warning("Falling back to opt_res_latest")
        else:
            logger.warning("No saved entity json files in path:" + str(entity_path))
            logger.warning("Falling back to opt_res_latest")
        filename = default_csv_filename
    else:
        filename = default_csv_filename
    if opt_res_latest is None:
        if not os.path.isfile(input_data_dict["emhass_conf"]["data_path"] / filename):
            logger.error("File not found error, run an optimization task first.")
            return
        else:
            opt_res_latest = pd.read_csv(
                input_data_dict["emhass_conf"]["data_path"] / filename,
                index_col="timestamp",
            )
            opt_res_latest.index = pd.to_datetime(opt_res_latest.index)
            opt_res_latest.index.freq = input_data_dict["retrieve_hass_conf"][
                "optimization_time_step"
            ]
    # Estimate the current index
    now_precise = datetime.now(
        input_data_dict["retrieve_hass_conf"]["time_zone"]
    ).replace(second=0, microsecond=0)
    if input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "nearest":
        idx_closest = opt_res_latest.index.get_indexer([now_precise], method="nearest")[
            0
        ]
    elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "first":
        idx_closest = opt_res_latest.index.get_indexer([now_precise], method="ffill")[0]
    elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "last":
        idx_closest = opt_res_latest.index.get_indexer([now_precise], method="bfill")[0]
    if idx_closest == -1:
        idx_closest = opt_res_latest.index.get_indexer([now_precise], method="nearest")[
            0
        ]
    # Publish the data
    publish_prefix = params["passed_data"]["publish_prefix"]
    # Publish PV forecast
    custom_pv_forecast_id = params["passed_data"]["custom_pv_forecast_id"]
    await input_data_dict["rh"].post_data(
        opt_res_latest["P_PV"],
        idx_closest,
        custom_pv_forecast_id["entity_id"],
        "power",
        custom_pv_forecast_id["unit_of_measurement"],
        custom_pv_forecast_id["friendly_name"],
        type_var="power",
        publish_prefix=publish_prefix,
        save_entities=entity_save,
        dont_post=dont_post,
    )
    # Publish Load forecast
    custom_load_forecast_id = params["passed_data"]["custom_load_forecast_id"]
    await input_data_dict["rh"].post_data(
        opt_res_latest["P_Load"],
        idx_closest,
        custom_load_forecast_id["entity_id"],
        "power",
        custom_load_forecast_id["unit_of_measurement"],
        custom_load_forecast_id["friendly_name"],
        type_var="power",
        publish_prefix=publish_prefix,
        save_entities=entity_save,
        dont_post=dont_post,
    )
    cols_published = ["P_PV", "P_Load"]
    # Publish PV curtailment
    if input_data_dict["fcst"].plant_conf["compute_curtailment"]:
        custom_pv_curtailment_id = params["passed_data"]["custom_pv_curtailment_id"]
        await input_data_dict["rh"].post_data(
            opt_res_latest["P_PV_curtailment"],
            idx_closest,
            custom_pv_curtailment_id["entity_id"],
            "power",
            custom_pv_curtailment_id["unit_of_measurement"],
            custom_pv_curtailment_id["friendly_name"],
            type_var="power",
            publish_prefix=publish_prefix,
            save_entities=entity_save,
            dont_post=dont_post,
        )
        cols_published = cols_published + ["P_PV_curtailment"]
    # Publish P_hybrid_inverter
    if input_data_dict["fcst"].plant_conf["inverter_is_hybrid"]:
        custom_hybrid_inverter_id = params["passed_data"]["custom_hybrid_inverter_id"]
        await input_data_dict["rh"].post_data(
            opt_res_latest["P_hybrid_inverter"],
            idx_closest,
            custom_hybrid_inverter_id["entity_id"],
            "power",
            custom_hybrid_inverter_id["unit_of_measurement"],
            custom_hybrid_inverter_id["friendly_name"],
            type_var="power",
            publish_prefix=publish_prefix,
            save_entities=entity_save,
            dont_post=dont_post,
        )
        cols_published = cols_published + ["P_hybrid_inverter"]
    # Publish deferrable loads
    custom_deferrable_forecast_id = params["passed_data"][
        "custom_deferrable_forecast_id"
    ]
    for k in range(input_data_dict["opt"].optim_conf["number_of_deferrable_loads"]):
        if f"P_deferrable{k}" not in opt_res_latest.columns:
            logger.error(
                f"P_deferrable{k}"
                + " was not found in results DataFrame. Optimization task may need to be relaunched or it did not converge to a solution.",
            )
        else:
            await input_data_dict["rh"].post_data(
                opt_res_latest[f"P_deferrable{k}"],
                idx_closest,
                custom_deferrable_forecast_id[k]["entity_id"],
                "power",
                custom_deferrable_forecast_id[k]["unit_of_measurement"],
                custom_deferrable_forecast_id[k]["friendly_name"],
                type_var="deferrable",
                publish_prefix=publish_prefix,
                save_entities=entity_save,
                dont_post=dont_post,
            )
            cols_published = cols_published + [f"P_deferrable{k}"]
    # Publish thermal model data (predicted temperature)
    custom_predicted_temperature_id = params["passed_data"][
        "custom_predicted_temperature_id"
    ]
    for k in range(input_data_dict["opt"].optim_conf["number_of_deferrable_loads"]):
        if "def_load_config" in input_data_dict["opt"].optim_conf.keys():
            if (
                "thermal_config"
                in input_data_dict["opt"].optim_conf["def_load_config"][k]
            ):
                await input_data_dict["rh"].post_data(
                    opt_res_latest[f"predicted_temp_heater{k}"],
                    idx_closest,
                    custom_predicted_temperature_id[k]["entity_id"],
                    "temperature",
                    custom_predicted_temperature_id[k]["unit_of_measurement"],
                    custom_predicted_temperature_id[k]["friendly_name"],
                    type_var="temperature",
                    publish_prefix=publish_prefix,
                    save_entities=entity_save,
                    dont_post=dont_post,
                )
                cols_published = cols_published + [f"predicted_temp_heater{k}"]
    # Publish battery power
    if input_data_dict["opt"].optim_conf["set_use_battery"]:
        if "P_batt" not in opt_res_latest.columns:
            logger.error(
                "P_batt was not found in results DataFrame. Optimization task may need to be relaunched or it did not converge to a solution.",
            )
        else:
            custom_batt_forecast_id = params["passed_data"]["custom_batt_forecast_id"]
            await input_data_dict["rh"].post_data(
                opt_res_latest["P_batt"],
                idx_closest,
                custom_batt_forecast_id["entity_id"],
                "power",
                custom_batt_forecast_id["unit_of_measurement"],
                custom_batt_forecast_id["friendly_name"],
                type_var="batt",
                publish_prefix=publish_prefix,
                save_entities=entity_save,
                dont_post=dont_post,
            )
            cols_published = cols_published + ["P_batt"]
            custom_batt_soc_forecast_id = params["passed_data"][
                "custom_batt_soc_forecast_id"
            ]
            await input_data_dict["rh"].post_data(
                opt_res_latest["SOC_opt"] * 100,
                idx_closest,
                custom_batt_soc_forecast_id["entity_id"],
                "battery",
                custom_batt_soc_forecast_id["unit_of_measurement"],
                custom_batt_soc_forecast_id["friendly_name"],
                type_var="SOC",
                publish_prefix=publish_prefix,
                save_entities=entity_save,
                dont_post=dont_post,
            )
            cols_published = cols_published + ["SOC_opt"]
    # Publish grid power
    custom_grid_forecast_id = params["passed_data"]["custom_grid_forecast_id"]
    await input_data_dict["rh"].post_data(
        opt_res_latest["P_grid"],
        idx_closest,
        custom_grid_forecast_id["entity_id"],
        "power",
        custom_grid_forecast_id["unit_of_measurement"],
        custom_grid_forecast_id["friendly_name"],
        type_var="power",
        publish_prefix=publish_prefix,
        save_entities=entity_save,
        dont_post=dont_post,
    )
    cols_published = cols_published + ["P_grid"]
    # Publish total value of cost function
    custom_cost_fun_id = params["passed_data"]["custom_cost_fun_id"]
    col_cost_fun = [i for i in opt_res_latest.columns if "cost_fun_" in i]
    await input_data_dict["rh"].post_data(
        opt_res_latest[col_cost_fun],
        idx_closest,
        custom_cost_fun_id["entity_id"],
        "monetary",
        custom_cost_fun_id["unit_of_measurement"],
        custom_cost_fun_id["friendly_name"],
        type_var="cost_fun",
        publish_prefix=publish_prefix,
        save_entities=entity_save,
        dont_post=dont_post,
    )
    # cols_published = cols_published + col_cost_fun
    # Publish the optimization status
    custom_optim_status_id = params["passed_data"]["custom_optim_status_id"]
    if "optim_status" not in opt_res_latest:
        opt_res_latest["optim_status"] = "Optimal"
        logger.warning(
            "no optim_status in opt_res_latest, run an optimization task first",
        )
    else:
        await input_data_dict["rh"].post_data(
            opt_res_latest["optim_status"],
            idx_closest,
            custom_optim_status_id["entity_id"],
            "",
            "",
            custom_optim_status_id["friendly_name"],
            type_var="optim_status",
            publish_prefix=publish_prefix,
            save_entities=entity_save,
            dont_post=dont_post,
        )
        cols_published = cols_published + ["optim_status"]
    # Publish unit_load_cost
    custom_unit_load_cost_id = params["passed_data"]["custom_unit_load_cost_id"]
    await input_data_dict["rh"].post_data(
        opt_res_latest["unit_load_cost"],
        idx_closest,
        custom_unit_load_cost_id["entity_id"],
        "monetary",
        custom_unit_load_cost_id["unit_of_measurement"],
        custom_unit_load_cost_id["friendly_name"],
        type_var="unit_load_cost",
        publish_prefix=publish_prefix,
        save_entities=entity_save,
        dont_post=dont_post,
    )
    cols_published = cols_published + ["unit_load_cost"]
    # Publish unit_prod_price
    custom_unit_prod_price_id = params["passed_data"]["custom_unit_prod_price_id"]
    await input_data_dict["rh"].post_data(
        opt_res_latest["unit_prod_price"],
        idx_closest,
        custom_unit_prod_price_id["entity_id"],
        "monetary",
        custom_unit_prod_price_id["unit_of_measurement"],
        custom_unit_prod_price_id["friendly_name"],
        type_var="unit_prod_price",
        publish_prefix=publish_prefix,
        save_entities=entity_save,
        dont_post=dont_post,
    )
    cols_published = cols_published + ["unit_prod_price"]
    # Create a DF resuming what has been published
    opt_res = opt_res_latest[cols_published].loc[[opt_res_latest.index[idx_closest]]]
    return opt_res


async def continual_publish(
    input_data_dict: dict, entity_path: pathlib.Path, logger: logging.Logger
):
    """
    If continual_publish is true and a entity file saved in /data_path/entities, continually publish sensor on freq rate, updating entity current state value based on timestamp

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param entity_path: Path for entities folder in data_path
    :type entity_path: Path
    :param logger: The passed logger object
    :type logger: logging.Logger

    """
    logger.info("Continual publish thread service started")
    freq = input_data_dict["retrieve_hass_conf"].get(
        "optimization_time_step", pd.to_timedelta(1, "minutes")
    )
    entity_path_contents = []
    while True:
        # Sleep for x seconds (using current time as a reference for time left)
        await asyncio.sleep(
            max(
                0,
                freq.total_seconds()
                - (
                    datetime.now(
                        input_data_dict["retrieve_hass_conf"]["time_zone"]
                    ).timestamp()
                    % 60
                ),
            )
        )
        # Loop through all saved entity files
        if os.path.exists(entity_path) and len(os.listdir(entity_path)) > 0:
            entity_path_contents = os.listdir(entity_path)
            for entity in entity_path_contents:
                if entity != default_metadata_json:
                    # Call publish_json with entity file, build entity, and publish
                    await publish_json(
                        entity,
                        input_data_dict,
                        entity_path,
                        logger,
                        "continual_publish",
                    )
            # Retrieve entity metadata from file
            if os.path.isfile(entity_path / default_metadata_json):
                async with aiofiles.open(entity_path / default_metadata_json) as file:
                    content = await file.read()
                    metadata = orjson.loads(content)
                    # Check if freq should be shorter
                    if metadata.get("lowest_time_step", None) is not None:
                        freq = pd.to_timedelta(metadata["lowest_time_step"], "minutes")
        pass
    # This function should never return
    return False


async def publish_json(
    entity: dict,
    input_data_dict: dict,
    entity_path: pathlib.Path,
    logger: logging.Logger,
    reference: str | None = "",
):
    """
    Extract saved entity data from .json (in data_path/entities), build entity, post results to post_data

    :param entity: json file containing entity data
    :type entity: dict
    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param entity_path: Path for entities folder in data_path
    :type entity_path: Path
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param reference: String for identifying who ran the function
    :type reference: str, optional

    """
    # Retrieve entity metadata from file
    if os.path.isfile(entity_path / default_metadata_json):
        async with aiofiles.open(entity_path / default_metadata_json) as file:
            content = await file.read()
            metadata = orjson.loads(content)
    else:
        logger.error("unable to located metadata.json in:" + entity_path)
        return False
    # Round current timecode (now)
    now_precise = datetime.now(
        input_data_dict["retrieve_hass_conf"]["time_zone"]
    ).replace(second=0, microsecond=0)
    # Retrieve entity data from file
    entity_data = pd.read_json(entity_path / entity, orient="index")
    # Remove ".json" from string for entity_id
    entity_id = entity.replace(".json", "")
    # Adjust Dataframe from received entity json file
    entity_data.columns = [metadata[entity_id]["name"]]
    entity_data.index.name = "timestamp"
    entity_data.index = pd.to_datetime(entity_data.index).tz_convert(
        input_data_dict["retrieve_hass_conf"]["time_zone"]
    )
    entity_data.index.freq = pd.to_timedelta(
        int(metadata[entity_id]["optimization_time_step"]), "minutes"
    )
    # Calculate the current state value
    if input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "nearest":
        idx_closest = entity_data.index.get_indexer([now_precise], method="nearest")[0]
    elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "first":
        idx_closest = entity_data.index.get_indexer([now_precise], method="ffill")[0]
    elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "last":
        idx_closest = entity_data.index.get_indexer([now_precise], method="bfill")[0]
    if idx_closest == -1:
        idx_closest = entity_data.index.get_indexer([now_precise], method="nearest")[0]
    # Call post data
    if reference == "continual_publish":
        logger.debug("Auto Published sensor:")
        logger_levels = "DEBUG"
    else:
        logger_levels = "INFO"
    # post/save entity
    await input_data_dict["rh"].post_data(
        data_df=entity_data[metadata[entity_id]["name"]],
        idx=idx_closest,
        entity_id=entity_id,
        device_class=dict.get(metadata[entity_id], "device_class"),
        unit_of_measurement=metadata[entity_id]["unit_of_measurement"],
        friendly_name=metadata[entity_id]["friendly_name"],
        type_var=metadata[entity_id].get("type_var", ""),
        save_entities=False,
        logger_levels=logger_levels,
    )
    return entity_data[metadata[entity_id]["name"]]


async def main():
    r"""Define the main command line entry function.

    This function may take several arguments as inputs. You can type `emhass --help` to see the list of options:

    - action: Set the desired action, options are: perfect-optim, dayahead-optim,
      naive-mpc-optim, publish-data, forecast-model-fit, forecast-model-predict, forecast-model-tune

    - config: Define path to the config.yaml file

    - costfun: Define the type of cost function, options are: profit, cost, self-consumption

    - log2file: Define if we should log to a file or not

    - params: Configuration parameters passed from data/options.json if using the add-on

    - runtimeparams: Pass runtime optimization parameters as dictionnary

    - debug: Use True for testing purposes

    """
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        type=str,
        help="Set the desired action, options are: perfect-optim, dayahead-optim,\
        naive-mpc-optim, publish-data, forecast-model-fit, forecast-model-predict, forecast-model-tune",
    )
    parser.add_argument(
        "--config", type=str, help="Define path to the config.json/defaults.json file"
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="String of configuration parameters passed",
    )
    parser.add_argument(
        "--data", type=str, help="Define path to the Data files (.csv & .pkl)"
    )
    parser.add_argument("--root", type=str, help="Define path emhass root")
    parser.add_argument(
        "--costfun",
        type=str,
        default="profit",
        help="Define the type of cost function, options are: profit, cost, self-consumption",
    )
    parser.add_argument(
        "--log2file",
        type=bool,
        default=False,
        help="Define if we should log to a file or not",
    )
    parser.add_argument(
        "--secrets",
        type=str,
        default=None,
        help="Define secret parameter file (secrets_emhass.yaml) path",
    )
    parser.add_argument(
        "--runtimeparams",
        type=str,
        default=None,
        help="Pass runtime optimization parameters as dictionnary",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Use True for testing purposes",
    )
    args = parser.parse_args()

    # The path to the configuration files
    if args.config is not None:
        config_path = pathlib.Path(args.config)
    else:
        config_path = pathlib.Path(
            str(utils.get_root(__file__, num_parent=3) / "config.json")
        )
    if args.data is not None:
        data_path = pathlib.Path(args.data)
    else:
        data_path = config_path.parent / "data/"
    if args.root is not None:
        root_path = pathlib.Path(args.root)
    else:
        root_path = utils.get_root(__file__, num_parent=1)
    if args.secrets is not None:
        secrets_path = pathlib.Path(args.secrets)
    else:
        secrets_path = pathlib.Path(config_path.parent / "secrets_emhass.yaml")

    associations_path = root_path / "data/associations.csv"
    defaults_path = root_path / "data/config_defaults.json"

    emhass_conf = {}
    emhass_conf["config_path"] = config_path
    emhass_conf["data_path"] = data_path
    emhass_conf["root_path"] = root_path
    emhass_conf["associations_path"] = associations_path
    emhass_conf["defaults_path"] = defaults_path
    # create logger
    logger, ch = utils.get_logger(
        __name__, emhass_conf, save_to_file=bool(args.log2file)
    )

    # Check paths
    logger.debug("config path: " + str(config_path))
    logger.debug("data path: " + str(data_path))
    logger.debug("root path: " + str(root_path))
    if not associations_path.exists():
        logger.error(
            "Could not find associations.csv file in: " + str(associations_path)
        )
        logger.error("Try setting config file path with --associations")
        return False
    if not config_path.exists():
        logger.warning("Could not find config.json file in: " + str(config_path))
        logger.warning("Try setting config file path with --config")
    if not secrets_path.exists():
        logger.warning("Could not find secrets file in: " + str(secrets_path))
        logger.warning("Try setting secrets file path with --secrets")
    if not os.path.isdir(data_path):
        logger.error("Could not find data folder in: " + str(data_path))
        logger.error("Try setting data path with --data")
        return False
    if not os.path.isdir(root_path):
        logger.error("Could not find emhass/src folder in: " + str(root_path))
        logger.error("Try setting emhass root path with --root")
        return False

    # Additional argument
    try:
        parser.add_argument(
            "--version",
            action="version",
            version="%(prog)s " + version("emhass"),
        )
        args = parser.parse_args()
    except Exception:
        logger.info(
            "Version not found for emhass package. Or importlib exited with PackageNotFoundError.",
        )

    # Setup config
    config = {}
    # Check if passed config file is yaml of json, build config accordingly
    if config_path.exists():
        config_file_ending = re.findall(r"(?<=\.).*$", str(config_path))
        if len(config_file_ending) > 0:
            match config_file_ending[0]:
                case "json":
                    config = await utils.build_config(
                        emhass_conf, logger, defaults_path, config_path
                    )
                case "yaml":
                    config = await utils.build_config(
                        emhass_conf, logger, defaults_path, config_path=config_path
                    )
                case "yml":
                    config = await utils.build_config(
                        emhass_conf, logger, defaults_path, config_path=config_path
                    )
    # If unable to find config file, use only defaults_config.json
    else:
        logger.warning(
            "Unable to obtain config.json file, building parameters with only defaults"
        )
        config = await utils.build_config(emhass_conf, logger, defaults_path)
    if type(config) is bool and not config:
        raise Exception("Failed to find default config")

    # Obtain secrets from secrets_emhass.yaml?
    params_secrets = {}
    emhass_conf, built_secrets = await utils.build_secrets(
        emhass_conf, logger, secrets_path=secrets_path
    )
    params_secrets.update(built_secrets)

    # Build params
    params = await utils.build_params(emhass_conf, params_secrets, config, logger)
    if type(params) is bool:
        raise Exception("A error has occurred while building parameters")
    # Add any passed params from args to params
    if args.params:
        params.update(orjson.loads(args.params))

    input_data_dict = await set_input_data_dict(
        emhass_conf,
        args.costfun,
        orjson.dumps(params).decode("utf-8"),
        args.runtimeparams,
        args.action,
        logger,
        args.debug,
    )
    if type(input_data_dict) is bool:
        raise Exception("A error has occurred while creating action objects")

    # Perform selected action
    if args.action == "perfect-optim":
        opt_res = await perfect_forecast_optim(input_data_dict, logger, debug=args.debug)
    elif args.action == "dayahead-optim":
        opt_res = await dayahead_forecast_optim(input_data_dict, logger, debug=args.debug)
    elif args.action == "naive-mpc-optim":
        opt_res = await naive_mpc_optim(input_data_dict, logger, debug=args.debug)
    elif args.action == "forecast-model-fit":
        df_fit_pred, df_fit_pred_backtest, mlf = forecast_model_fit(
            input_data_dict, logger, debug=args.debug
        )
        opt_res = None
    elif args.action == "forecast-model-predict":
        if args.debug:
            _, _, mlf = await forecast_model_fit(input_data_dict, logger, debug=args.debug)
        else:
            mlf = None
        df_pred = await forecast_model_predict(
            input_data_dict, logger, debug=args.debug, mlf=mlf
        )
        opt_res = None
    elif args.action == "forecast-model-tune":
        if args.debug:
            _, _, mlf = await forecast_model_fit(input_data_dict, logger, debug=args.debug)
        else:
            mlf = None
        df_pred_optim, mlf = await forecast_model_tune(
            input_data_dict, logger, debug=args.debug, mlf=mlf
        )
        opt_res = None
    elif args.action == "regressor-model-fit":
        mlr = await regressor_model_fit(input_data_dict, logger, debug=args.debug)
        opt_res = None
    elif args.action == "regressor-model-predict":
        if args.debug:
            mlr = await regressor_model_fit(input_data_dict, logger, debug=args.debug)
        else:
            mlr = None
        prediction = await regressor_model_predict(
            input_data_dict, logger, debug=args.debug, mlr=mlr
        )
        opt_res = None
    elif args.action == "publish-data":
        opt_res = await publish_data(input_data_dict, logger)
    else:
        logger.error("The passed action argument is not valid")
        logger.error(
            "Try setting --action: perfect-optim, dayahead-optim, naive-mpc-optim, forecast-model-fit, forecast-model-predict, forecast-model-tune or publish-data"
        )
        opt_res = None
    logger.info(opt_res)
    # Flush the logger
    ch.close()
    logger.removeHandler(ch)
    if (
        args.action == "perfect-optim"
        or args.action == "dayahead-optim"
        or args.action == "naive-mpc-optim"
        or args.action == "publish-data"
    ):
        return opt_res
    elif args.action == "forecast-model-fit":
        return df_fit_pred, df_fit_pred_backtest, mlf
    elif args.action == "forecast-model-predict":
        return df_pred
    elif args.action == "regressor-model-fit":
        return mlr
    elif args.action == "regressor-model-predict":
        return prediction
    elif args.action == "forecast-model-tune":
        return df_pred_optim, mlf
    else:
        return opt_res


def main_sync():
    """Sync wrapper for async main function - used as CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
