import csv
import re

with open('./DATA/Squiggles(833)/COMPLETE_squiggle_dft.csv', 'rU') as csvinput:
    with open('./DATA/Squiggles(833)/COMPLETE_squiggle_dft_PARSED.csv', 'w') as csvoutput:
            reader = csv.DictReader(csvinput)
            fieldnames = ['year', 'month', 'day', 'time_of_day', 'timezone', 'activityID', 'correlatorID', 'beam_number', 'polarization'] + reader.fieldnames

            writer = csv.DictWriter(csvoutput, fieldnames = fieldnames)
            writer.writeheader()

            rows = []
            for row in reader:
            	idString = row['id']
            	m = re.match(r'(\d\d\d\d)-(\d\d)-(\d\d)_(\d\d)-(\d\d)-(\d\d)_(\w+).(\w+).(\w+).(\w*).(\w*).(\w*).(png)', idString)
            	year = m.group(1)
            	month = m.group(2)
            	day = m.group(3)
            	hour = m.group(4)
            	minute = m.group(5)
            	second = m.group(6)
            	timezone = m.group(7)
            	activityID = m.group(8)
            	correlatorID = m.group(9)
            	beam_number = m.group(10)
            	polarization = m.group(11)

                row['year'] = year
                row['month'] = month
                row['day'] = day
                row['time_of_day'] = float(hour) + (1./60) * (float(minute)) + (1./3600) * (float(second))
                row['timezone'] = timezone
                row['activityID'] = activityID
                row['correlatorID'] = correlatorID
                row['beam_number'] = beam_number
                row['polarization'] = polarization

                rows.append(row)

            writer.writerows(rows)

#(\d\d\d\d)-(\d\d)-(\d\d)_(\d\d)-(\d\d)-(\d\d)_(\w+).(\w+).(\w+).(.*).(.*)(\.png)