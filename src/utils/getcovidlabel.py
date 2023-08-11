def getCovidLabel(caf_id,full_config):

    current_row = full_config[full_config['FILENAME'] == caf_id + '.caf']
    if len(current_row) > 0:
        return current_row['covid'].iat[0]
    else:
        print('No apparent label for: ' + str(caf_id))
        return 'NA'
